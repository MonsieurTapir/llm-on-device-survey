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
# --ignore-whitespace because line endings are not ours to control on both sides:
# this repo and the fetched llama.cpp are two checkouts under whatever
# core.autocrlf the machine has, and llama.cpp pins none of its own. Matching
# context lines across a CRLF/LF mismatch is the whole reason for the flag; the
# added lines still go in verbatim, and a patch's code is what the reverse-check
# looks for, so tolerating whitespace cannot fake "already applied".
#
# FetchContent runs the populate step quietly, so the STATUS lines below only
# reach the terminal with -DFETCHCONTENT_QUIET=OFF; otherwise they land in
# _deps/llamacpp-subbuild/.../llamacpp-populate-*.log. A FATAL_ERROR always
# surfaces — a failed populate prints its whole log.
cmake_minimum_required(VERSION 3.18)

foreach(patch IN LISTS PATCHES)
  get_filename_component(name "${patch}" NAME)
  execute_process(COMMAND "${GIT}" apply --reverse --check --ignore-whitespace "${patch}"
                  WORKING_DIRECTORY "${SRC}"
                  RESULT_VARIABLE applied
                  OUTPUT_QUIET ERROR_QUIET)
  if(applied EQUAL 0)
    message(STATUS "llama.cpp patch already applied: ${name}")
    continue()
  endif()

  execute_process(COMMAND "${GIT}" apply --ignore-whitespace "${patch}"
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
