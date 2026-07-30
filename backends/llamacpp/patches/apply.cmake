# Idempotent patch application for the fetched llama.cpp tree — FetchContent's
# PATCH_COMMAND (see ../CMakeLists.txt). Invoked as:
#   cmake -DGIT=<git> -DSRC=<tree> -DPATCHES=<;-separated list> -P apply.cmake
#
# Each patch is reverse-checked before it is applied: `git apply --reverse
# --check` succeeds only when the patch is already in the tree, so a re-run is a
# no-op rather than a "patch does not apply" hard error. A patch that neither
# applies nor reverse-applies is a real conflict and fails the configure —
# silently building an unpatched stack would be worse.
#
# FetchContent runs the populate step quietly, so the STATUS lines below only
# reach the terminal with -DFETCHCONTENT_QUIET=OFF; otherwise they land in
# _deps/llamacpp-subbuild/.../llamacpp-populate-*.log. A FATAL_ERROR always
# surfaces — a failed populate prints its whole log.
cmake_minimum_required(VERSION 3.18)

foreach(patch IN LISTS PATCHES)
  get_filename_component(name "${patch}" NAME)
  execute_process(COMMAND "${GIT}" apply --reverse --check "${patch}"
                  WORKING_DIRECTORY "${SRC}"
                  RESULT_VARIABLE applied
                  OUTPUT_QUIET ERROR_QUIET)
  if(applied EQUAL 0)
    message(STATUS "llama.cpp patch already applied: ${name}")
    continue()
  endif()

  execute_process(COMMAND "${GIT}" apply "${patch}"
                  WORKING_DIRECTORY "${SRC}"
                  RESULT_VARIABLE result
                  ERROR_VARIABLE stderr)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR
      "llama.cpp patch does not apply: ${name}\n${stderr}"
      "Either the pinned commit moved past it or the tree is dirty. "
      "Re-check the patch against LLAMACPP_GIT_TAG, or drop it if upstream fixed it.")
  endif()
  message(STATUS "llama.cpp patched: ${name}")
endforeach()
