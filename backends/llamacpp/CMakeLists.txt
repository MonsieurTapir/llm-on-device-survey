# survey-llamacpp — the llama.cpp backend exe.
#
# Dependencies are self-contained: both llama.cpp and CLI11 are fetched at
# configure time, pinned for reproducibility — no top-level third_party/ checkout
# is required. To build against an existing llama.cpp instead (e.g. offline, or a
# local working copy), point LLAMACPP_SOURCE_DIR at it:
#   cmake -B build -S . -DLLAMACPP_SOURCE_DIR=../../third_party/llama.cpp
#
# Produces ./build/survey-llamacpp — the path declared in backend.toml.
# Metal is auto-enabled on macOS; CUDA/Vulkan are opt-in:
#   cmake -B build -S .                          # CPU + Metal on macOS, CPU-only elsewhere
#   cmake -B build -S . -DGGML_METAL=OFF         # force CPU-only on macOS
#   cmake -B build -S . -DGGML_CUDA=ON
cmake_minimum_required(VERSION 3.18)  # DOWNLOAD_NO_EXTRACT (CLI11 single header)
project(survey-llamacpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Metal is always available on macOS (10.11+); default ON so a plain
# `cmake -B build -S .` gives both cpu and metal providers on Mac.
#
# GGML_METAL_EMBED_LIBRARY is pinned rather than inherited: ggml defaults it to
# ${GGML_METAL}, and it decides whether the Metal shader source is embedded and
# compiled in-process or shipped beside the exe as a prebuilt default.metallib.
# That moves real cost between build time and the measured load phases, and with
# it OFF a missing metallib does not fail — ggml silently falls back to compiling
# from source. Pinning it keeps every Mac submission measuring the same stack even
# if the upstream default changes; the resolved value is echoed by `version`.
if(APPLE)
  option(GGML_METAL "Metal GPU support" ON)
  set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()

include(FetchContent)

# --- llama.cpp ---------------------------------------------------------------
# Pinned to the commit this backend is developed against. Override with
# -DLLAMACPP_SOURCE_DIR=<path> to consume a local checkout instead of fetching.
set(LLAMACPP_GIT_TAG "1c5b89ff63151b5388cb5abb9357cebe446997c7" CACHE STRING
    "llama.cpp commit to fetch when LLAMACPP_SOURCE_DIR is unset")
set(LLAMACPP_SOURCE_DIR "" CACHE PATH
    "Path to an existing llama.cpp checkout; if empty, the pinned commit is fetched")

# We embed llama.cpp as a subproject. It defaults most of these to its
# LLAMA_STANDALONE value (OFF when embedded), so force the ones we need BEFORE it
# is added to the build:
#   - COMMON ON   : we use common_tokenize / build-info / vendored nlohmann json.
#   - the rest OFF: we don't want its tests/tools/examples/server, and we don't
#                   want the OpenSSL/cURL deps that the server/common HTTP bits pull.
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI       OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)

if(LLAMACPP_SOURCE_DIR)
  message(STATUS "survey-llamacpp: using local llama.cpp at ${LLAMACPP_SOURCE_DIR}")
  add_subdirectory(${LLAMACPP_SOURCE_DIR} ${CMAKE_BINARY_DIR}/llama.cpp)
  set(LLAMACPP_SRC_INTERNAL "${LLAMACPP_SOURCE_DIR}/src")
else()
  message(STATUS "survey-llamacpp: fetching llama.cpp @ ${LLAMACPP_GIT_TAG}")
  # Local fixes to the pinned tree, applied in listed order. Each patch file
  # documents the bug it fixes and what would replace it upstream; every one is a
  # deliberate delta on the compatibility anchor, so keep the list short and
  # delete entries as the pin moves past them.
  #
  # These apply to the *fetched* tree only. A LLAMACPP_SOURCE_DIR checkout is
  # yours to manage — we don't write into a working copy we didn't create, so a
  # local-checkout build carries whatever that tree already has.
  #
  # FetchContent patches at populate time only: after editing anything under
  # patches/, delete the build directory so the tree is re-populated.
  set(LLAMACPP_PATCHES
    "${CMAKE_CURRENT_SOURCE_DIR}/patches/0001-vulkan-gate-coopmat-bf16-on-the-bfloat16-extension.patch")
  find_package(Git REQUIRED)
  FetchContent_Declare(llamacpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        ${LLAMACPP_GIT_TAG}
    PATCH_COMMAND  ${CMAKE_COMMAND} -DGIT=${GIT_EXECUTABLE} -DSRC=<SOURCE_DIR>
                   "-DPATCHES=${LLAMACPP_PATCHES}"
                   -P "${CMAKE_CURRENT_SOURCE_DIR}/patches/apply.cmake")
  FetchContent_MakeAvailable(llamacpp)
  set(LLAMACPP_SRC_INTERNAL "${llamacpp_SOURCE_DIR}/src")
endif()

# --- CLI11 (single header) ---------------------------------------------------
# Pinned by content hash; DOWNLOAD_NO_EXTRACT leaves the bare header in the dir.
FetchContent_Declare(cli11
  URL      https://github.com/CLIUtils/CLI11/releases/download/v2.7.0/CLI11.hpp
  URL_HASH SHA256=5aa8d6378792163d1e0df642eee313aceeca68a53464566ae7204d59c2f5e168
  DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_MakeAvailable(cli11)

# --- the executable ----------------------------------------------------------
add_executable(survey-llamacpp main.cpp)

# Linking llama-common transitively brings llama (→ ggml), its include dirs, the
# common helpers, build-info, and the vendored nlohmann/json.hpp. cli11_SOURCE_DIR
# holds the downloaded CLI11.hpp. llama.cpp's src/ is on the path for
# llama-ext.h (llama_get_memory_breakdown — LLAMA_API-exported but declared
# outside the public include dir; same consumption pattern as common/fit.cpp).
target_link_libraries(survey-llamacpp PRIVATE llama llama-common)
target_include_directories(survey-llamacpp PRIVATE ${cli11_SOURCE_DIR} ${LLAMACPP_SRC_INTERNAL})

# ggml scopes its own GGML_METAL_EMBED_LIBRARY define to the ggml-metal directory,
# so hand the resolved value to our translation unit for `version` to report.
if(APPLE)
  target_compile_definitions(survey-llamacpp PRIVATE
    BENCH_METAL_EMBED_LIBRARY=$<BOOL:${GGML_METAL_EMBED_LIBRARY}>)
endif()

# Ad-hoc sign with get-task-allow so the survey tool can call task_for_pid and read
# task_vm_info (full memory decomposition including Metal/IOKit allocations).
if(APPLE)
  add_custom_command(TARGET survey-llamacpp POST_BUILD
    COMMAND codesign --entitlements ${CMAKE_CURRENT_SOURCE_DIR}/entitlements.plist
                     -fs - $<TARGET_FILE:survey-llamacpp>
    COMMENT "ad-hoc signing survey-llamacpp with get-task-allow entitlement")
endif()
