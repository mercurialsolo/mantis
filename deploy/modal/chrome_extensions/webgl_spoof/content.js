(function () {
  "use strict";

  // Linux-appropriate renderer strings — consistent with
  // navigator.platform === "Linux x86_64" inside the container.
  var SPOOFED_VENDOR = "Intel";
  var SPOOFED_RENDERER =
    "ANGLE (Intel, Mesa Intel(R) UHD Graphics 630 (CFL GT2), OpenGL 4.5)";

  // WEBGL_debug_renderer_info extension constants
  var UNMASKED_VENDOR_WEBGL = 0x9245;
  var UNMASKED_RENDERER_WEBGL = 0x9246;

  function patchContext(CtxClass) {
    var origGetParameter = CtxClass.prototype.getParameter;
    if (!origGetParameter) return;

    CtxClass.prototype.getParameter = function (param) {
      if (param === UNMASKED_RENDERER_WEBGL) return SPOOFED_RENDERER;
      if (param === UNMASKED_VENDOR_WEBGL) return SPOOFED_VENDOR;
      return origGetParameter.call(this, param);
    };
  }

  if (typeof WebGLRenderingContext !== "undefined")
    patchContext(WebGLRenderingContext);
  if (typeof WebGL2RenderingContext !== "undefined")
    patchContext(WebGL2RenderingContext);

  // Ensure navigator.languages is populated
  try {
    if (!navigator.languages || navigator.languages.length === 0) {
      Object.defineProperty(navigator, "languages", {
        get: function () {
          return ["en-US", "en"];
        },
      });
    }
  } catch (e) {}
})();
