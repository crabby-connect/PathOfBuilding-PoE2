//! Minimal LuaJIT (Lua 5.1 ABI) binding, resolved at runtime from the shipped
//! `runtime/lua51.dll`. Extends the PoC binding with what the production worker
//! needs: push an integer array (a candidate node-id set) as a Lua table and
//! call a worker function that takes that table and returns a number.
//!
//! We bind only the handful of C-API entry points we use, looked up by name
//! from the same DLL the host uses (no import lib, no headers). LuaJIT keeps no
//! shared mutable state across `lua_State`s, so distinct states created from the
//! same DLL run fully independently — one per worker thread (LuaJIT's
//! one-state-per-thread rule).

use libloading::{Library, Symbol};
use std::ffi::{c_char, c_double, c_int, c_void, CString};
use std::ptr;

pub type LuaState = *mut c_void;

// Lua 5.1 constants we use.
const LUA_GLOBALSINDEX: c_int = -10002;
const LUA_MULTRET: c_int = -1;
pub const LUA_OK: c_int = 0;
const LUA_TNUMBER: c_int = 3;

// C-API function pointer types (Lua 5.1 / LuaJIT cdecl).
type FnLNewState = unsafe extern "C" fn() -> LuaState;
type FnOpenLibs = unsafe extern "C" fn(l: LuaState);
type FnClose = unsafe extern "C" fn(l: LuaState);
type FnLoadString = unsafe extern "C" fn(l: LuaState, s: *const c_char) -> c_int;
type FnPCall =
    unsafe extern "C" fn(l: LuaState, nargs: c_int, nresults: c_int, errfunc: c_int) -> c_int;
type FnGetField = unsafe extern "C" fn(l: LuaState, idx: c_int, k: *const c_char);
type FnToLString = unsafe extern "C" fn(l: LuaState, idx: c_int, len: *mut usize) -> *const c_char;
type FnToNumber = unsafe extern "C" fn(l: LuaState, idx: c_int) -> c_double;
type FnSetTop = unsafe extern "C" fn(l: LuaState, idx: c_int);
type FnType = unsafe extern "C" fn(l: LuaState, idx: c_int) -> c_int;
type FnCreateTable = unsafe extern "C" fn(l: LuaState, narr: c_int, nrec: c_int);
type FnPushNumber = unsafe extern "C" fn(l: LuaState, n: c_double);
type FnRawSetI = unsafe extern "C" fn(l: LuaState, idx: c_int, n: c_int);
type FnObjLen = unsafe extern "C" fn(l: LuaState, idx: c_int) -> usize;
type FnRawGetI = unsafe extern "C" fn(l: LuaState, idx: c_int, n: c_int);

/// Loaded once per process; the DLL is reused by every worker state.
pub struct LuaLib {
    _lib: Library,
    l_newstate: FnLNewState,
    open_libs: FnOpenLibs,
    close: FnClose,
    load_string: FnLoadString,
    pcall: FnPCall,
    get_field: FnGetField,
    to_lstring: FnToLString,
    to_number: FnToNumber,
    set_top: FnSetTop,
    ty: FnType,
    create_table: FnCreateTable,
    push_number: FnPushNumber,
    raw_set_i: FnRawSetI,
    obj_len: FnObjLen,
    raw_get_i: FnRawGetI,
}

// The function pointers are plain code addresses in a leaked-for-lifetime DLL;
// sharing the LuaLib across threads is sound (each thread uses its own state).
unsafe impl Send for LuaLib {}
unsafe impl Sync for LuaLib {}

impl LuaLib {
    pub unsafe fn load(dll_path: &str) -> Result<Self, String> {
        let lib = Library::new(dll_path).map_err(|e| format!("load {dll_path}: {e}"))?;
        macro_rules! sym {
            ($name:literal, $t:ty) => {{
                let s: Symbol<$t> = lib
                    .get($name)
                    .map_err(|e| format!("symbol {}: {e}", String::from_utf8_lossy($name)))?;
                *s.into_raw()
            }};
        }
        Ok(LuaLib {
            l_newstate: sym!(b"luaL_newstate", FnLNewState),
            open_libs: sym!(b"luaL_openlibs", FnOpenLibs),
            close: sym!(b"lua_close", FnClose),
            load_string: sym!(b"luaL_loadstring", FnLoadString),
            pcall: sym!(b"lua_pcall", FnPCall),
            get_field: sym!(b"lua_getfield", FnGetField),
            to_lstring: sym!(b"lua_tolstring", FnToLString),
            to_number: sym!(b"lua_tonumber", FnToNumber),
            set_top: sym!(b"lua_settop", FnSetTop),
            ty: sym!(b"lua_type", FnType),
            create_table: sym!(b"lua_createtable", FnCreateTable),
            push_number: sym!(b"lua_pushnumber", FnPushNumber),
            raw_set_i: sym!(b"lua_rawseti", FnRawSetI),
            // luaL_newstate aliases aside, lua_objlen is the 5.1 name (LuaJIT keeps it).
            obj_len: sym!(b"lua_objlen", FnObjLen),
            raw_get_i: sym!(b"lua_rawgeti", FnRawGetI),
            _lib: lib,
        })
    }
}

/// One LuaJIT interpreter state. Created and used by a single thread for its
/// whole life (LuaJIT's one-state-per-thread rule).
pub struct Lua<'a> {
    lib: &'a LuaLib,
    l: LuaState,
}

impl<'a> Lua<'a> {
    pub unsafe fn new(lib: &'a LuaLib) -> Result<Self, String> {
        let l = (lib.l_newstate)();
        if l.is_null() {
            return Err("luaL_newstate returned null (out of memory)".into());
        }
        (lib.open_libs)(l);
        Ok(Lua { lib, l })
    }

    /// Load + run a chunk with no args, no results kept.
    pub unsafe fn run(&self, chunk: &str) -> Result<(), String> {
        let c = CString::new(chunk).map_err(|_| "chunk has NUL".to_string())?;
        if (self.lib.load_string)(self.l, c.as_ptr()) != LUA_OK {
            return Err(self.pop_error("load"));
        }
        if (self.lib.pcall)(self.l, 0, LUA_MULTRET, 0) != LUA_OK {
            return Err(self.pop_error("run"));
        }
        (self.lib.set_top)(self.l, 0);
        Ok(())
    }

    /// Call `name(ids_table)` where `ids_table` is a 1-based Lua array of the
    /// given integer node ids, expecting a single number back. This is the hot
    /// path: one candidate node-set -> one score.
    pub unsafe fn call_global_score(&self, name: &str, ids: &[i32]) -> Result<f64, String> {
        self.push_global(name)?;
        // Build the array arg: createtable(narr=len, nrec=0), then rawseti 1..n.
        (self.lib.create_table)(self.l, ids.len() as c_int, 0);
        for (i, &id) in ids.iter().enumerate() {
            (self.lib.push_number)(self.l, id as c_double);
            // rawseti pops the value and sets t[i+1] = value; table is now at -2.
            (self.lib.raw_set_i)(self.l, -2, (i + 1) as c_int);
        }
        if (self.lib.pcall)(self.l, 1, 1, 0) != LUA_OK {
            return Err(self.pop_error(name));
        }
        self.pop_number(name)
    }

    /// Call `name(ids_table)` expecting THREE numbers back: (score, dps, ehp).
    /// Same arg-building as call_global_score; used by the clean-slate path, which
    /// must carry dps/ehp so the host's Pareto beam can prune by both axes.
    pub unsafe fn call_global_score3(&self, name: &str, ids: &[i32]) -> Result<[f64; 3], String> {
        self.push_global(name)?;
        (self.lib.create_table)(self.l, ids.len() as c_int, 0);
        for (i, &id) in ids.iter().enumerate() {
            (self.lib.push_number)(self.l, id as c_double);
            (self.lib.raw_set_i)(self.l, -2, (i + 1) as c_int);
        }
        // nresults = 3: Lua pads with nil if the fn returns fewer; we validate types.
        if (self.lib.pcall)(self.l, 1, 3, 0) != LUA_OK {
            return Err(self.pop_error(name));
        }
        // Stack (bottom->top): score(-3), dps(-2), ehp(-1). Read before popping.
        let mut out = [0.0f64; 3];
        for (slot, idx) in [(0usize, -3i32), (1, -2), (2, -1)] {
            if (self.lib.ty)(self.l, idx) != LUA_TNUMBER {
                (self.lib.set_top)(self.l, 0);
                return Err(format!("{name}: result {} is not a number", slot + 1));
            }
            out[slot] = (self.lib.to_number)(self.l, idx);
        }
        (self.lib.set_top)(self.l, 0);
        Ok(out)
    }

    /// Call a zero-arg global function expected to return a Lua array of numbers
    /// (e.g. `__pob_candidate_ids`). Returns the values truncated to i32. Used
    /// once at setup, not on the hot path.
    pub unsafe fn call_global_int_array(&self, name: &str) -> Result<Vec<i32>, String> {
        self.push_global(name)?;
        if (self.lib.pcall)(self.l, 0, 1, 0) != LUA_OK {
            return Err(self.pop_error(name));
        }
        // 5 == LUA_TTABLE in 5.1.
        const LUA_TTABLE: c_int = 5;
        if (self.lib.ty)(self.l, -1) != LUA_TTABLE {
            (self.lib.set_top)(self.l, 0);
            return Err(format!("{name}: result is not a table"));
        }
        let len = (self.lib.obj_len)(self.l, -1);
        let mut out = Vec::with_capacity(len);
        for i in 1..=len {
            (self.lib.raw_get_i)(self.l, -1, i as c_int); // pushes t[i]
            let v = (self.lib.to_number)(self.l, -1);
            (self.lib.set_top)(self.l, -2); // pop the element, keep the table
            out.push(v as i32);
        }
        (self.lib.set_top)(self.l, 0);
        Ok(out)
    }

    /// Push a global function (by name) onto the stack, erroring if it is not a
    /// function — guards against silently calling nil.
    unsafe fn push_global(&self, name: &str) -> Result<(), String> {
        let c = CString::new(name).map_err(|_| "name has NUL".to_string())?;
        (self.lib.get_field)(self.l, LUA_GLOBALSINDEX, c.as_ptr());
        // 6 == LUA_TFUNCTION in 5.1; check positively to give a clear error.
        const LUA_TFUNCTION: c_int = 6;
        if (self.lib.ty)(self.l, -1) != LUA_TFUNCTION {
            (self.lib.set_top)(self.l, 0);
            return Err(format!("global '{name}' is not a function"));
        }
        Ok(())
    }

    /// Pop the top-of-stack result, requiring it be a number. Clears the stack.
    unsafe fn pop_number(&self, ctx: &str) -> Result<f64, String> {
        if (self.lib.ty)(self.l, -1) != LUA_TNUMBER {
            (self.lib.set_top)(self.l, 0);
            return Err(format!("{ctx}: result is not a number"));
        }
        let n = (self.lib.to_number)(self.l, -1);
        (self.lib.set_top)(self.l, 0);
        Ok(n)
    }

    unsafe fn pop_error(&self, ctx: &str) -> String {
        let mut len: usize = 0;
        let p = (self.lib.to_lstring)(self.l, -1, &mut len);
        let msg = if p.is_null() {
            "<no error message>".to_string()
        } else {
            let bytes = std::slice::from_raw_parts(p as *const u8, len);
            String::from_utf8_lossy(bytes).into_owned()
        };
        (self.lib.set_top)(self.l, 0);
        format!("{ctx}: {msg}")
    }
}

impl<'a> Drop for Lua<'a> {
    fn drop(&mut self) {
        unsafe { (self.lib.close)(self.l) };
        self.l = ptr::null_mut();
    }
}
