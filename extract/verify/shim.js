// Minimal Turbopack-module-system shim so the game's own physics chunk can
// be run standalone under Node, purely for local fidelity verification
// (see tests/test_fidelity_vs_real_bundle.py). It does not change, patch,
// or re-host the chunk's logic -- it only intercepts the two calls the
// chunk's factories make to register/export things (`t.s`) and to import
// another module (`t.i`, stubbed to {} since the physics chunk doesn't
// depend on any other chunk).
const path = require("path");

function loadPhysicsExports(chunkPath) {
  const exportsOut = {};
  const previousTurbopack = globalThis.TURBOPACK;
  globalThis.TURBOPACK = {
    push(arr) {
      for (let i = 1; i < arr.length; i++) {
        if (typeof arr[i] === "function") {
          const factory = arr[i];
          const t = {
            i: () => ({}),
            s: (pairs) => {
              for (let k = 0; k < pairs.length; k += 3) exportsOut[pairs[k]] = pairs[k + 2];
            },
            v: () => {},
          };
          try {
            factory(t);
          } catch (e) {
            // Other, unrelated modules bundled in the same chunk may
            // reference browser globals (document/localStorage) that
            // don't exist under Node -- harmless for our purposes.
          }
        }
      }
    },
  };
  delete require.cache[require.resolve(path.resolve(chunkPath))];
  require(path.resolve(chunkPath));
  globalThis.TURBOPACK = previousTurbopack;
  return exportsOut;
}

module.exports = { loadPhysicsExports };
