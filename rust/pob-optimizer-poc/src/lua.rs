//! Minimal LuaJIT (Lua 5.1 ABI) binding, resolved at runtime from the shipped
//! `runtime/lua51.dll`. We bind only the handful of C-API entry points the PoC
//! needs: create a state, open libs, load+run a chunk, call a global, read a
//! number/string result. No import lib, no headers — symbols are looked up by
//! name from the same DLL the host uses, which is also the production model.

use libloading::{Library, Symbol};
use std::ffi::{c_char, c_double, c_int, c_void, CString};
use std::ptr;

pub type LuaState = *mut c_void;

// Lua 5.1 constants we use.
const LUA_GLOBALSINDEX: c_int = -10002;
const LUA_MULTRET: c_int = -1;
pub const LUA_OK: c_int = 0;

// C-API function pointer types (Lua 5.1 / LuaJIT cdecl).
type FnNewState = unsafe extern "C" fn(alloc: *const c_void, ud: *const c_void) -> LuaState;
type FnLNewState = unsafe extern "C" fn() -> LuaState;
type FnOpenLibs = unsafe extern "C" fn(l: LuaState);
type FnClose = unsafe extern "C" fn(l: LuaState);
type FnLoadString = unsafe extern "C" fn(l: LuaState, s: *const c_char) -> c_int;
type FnPCall = unsafe extern "C" fn(l: LuaState, nargs: c_int, nresults: c_int, errfunc: c_int) -> c_int;
type FnGetField = unsafe extern "C" fn(l: LuaState, idx: c_int, k: *const c_char);
type FnToLString = unsafe extern "C" fn(l: LuaState, idx: c_int, len: *mut usize) -> *const c_char;
type FnToNumber = unsafe extern "C" fn(l: LuaState, idx: c_int) -> c_double;
type FnSetTop = unsafe extern "C" fn(l: LuaState, idx: c_int);
type FnType = unsafe extern "C" fn(l: LuaState, idx: c_int) -> c_int;

/// Loaded once per process; the DLL is reused by every worker state. LuaJIT
/// keeps no shared mutable state across `lua_State`s, so distinct states from
/// the same DLL run fully independently on different threads.
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
        // luaL_newstate exists on LuaJIT; fall back is not needed here.
        let l_newstate = sym!(b"luaL_newstate", FnLNewState);
        Ok(LuaLib {
            l_newstate,
            open_libs: sym!(b"luaL_openlibs", FnOpenLibs),
            close: sym!(b"lua_close", FnClose),
            load_string: sym!(b"luaL_loadstring", FnLoadString),
            pcall: sym!(b"lua_pcall", FnPCall),
            get_field: sym!(b"lua_getfield", FnGetField),
            to_lstring: sym!(b"lua_tolstring", FnToLString),
            to_number: sym!(b"lua_tonumber", FnToNumber),
            set_top: sym!(b"lua_settop", FnSetTop),
            ty: sym!(b"lua_type", FnType),
            _lib: lib,
        })
    }
}

/// One LuaJIT interpreter state. Created and used by a single thread.
pub struct Lua<'a> {
    lib: &'a LuaLib,
    l: LuaState,
}

impl<'a> Lua<'a> {
    pub unsafe fn new(lib: &'a LuaLib) -> Result<Self, String> {
        let l = (lib.l_newstate)();
        if l.is_null() {
            return Err("luaL_newstate returned null".into());
        }
        (lib.open_libs)(l);
        Ok(Lua { lib, l })
    }

    /// Load + run a chunk with no args, no results kept. Returns the Lua error
    /// string on failure.
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

    /// Call a zero-arg global function expected to return a single number.
    /// Used as the hot path: `__poc_eval()` -> score.
    pub unsafe fn call_global_number(&self, name: &str) -> Result<f64, String> {
        let c = CString::new(name).unwrap();
        (self.lib.get_field)(self.l, LUA_GLOBALSINDEX, c.as_ptr());
        if (self.lib.pcall)(self.l, 0, 1, 0) != LUA_OK {
            return Err(self.pop_error(name));
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

    pub unsafe fn type_at(&self, idx: c_int) -> c_int {
        (self.lib.ty)(self.l, idx)
    }
}

impl<'a> Drop for Lua<'a> {
    fn drop(&mut self) {
        unsafe { (self.lib.close)(self.l) };
        self.l = ptr::null_mut();
    }
}
