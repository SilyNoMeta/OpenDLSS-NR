#include "exec.h"

namespace exec {

const char* backendName(Backend backend) {
  switch (backend) {
    case Backend::Native: return "native";
    case Backend::Compat: return "compat";
    case Backend::Sm86: return "sm86";
  }
  return "?";
}

const char* backendDescription(Backend backend) {
  switch (backend) {
    case Backend::Native: return "native FP8 (E4M3 cooperative matrices and FP8 PTX: Ada, Hopper, Blackwell)";
    case Backend::Compat: return "compatibility (software E4M3 in scalar GLSL, no tensor cores, no fusion, no chaining)";
    case Backend::Sm86:
      return "sm86 (native PTX lowered to f16 tensor cores, compatibility GLSL elsewhere; not bit-exact to native)";
  }
  return "?";
}

}  // namespace exec
