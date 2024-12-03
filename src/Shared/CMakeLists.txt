# mepo can now clone subrepos in three styles
set (GMAOPYOBS_DIRS
  GMAOpyobs
  @GMAOpyobs
  GMAOpyobs@
)
foreach (dir IN LISTS GMAOPYOBS_DIRS)
  if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/${dir})
    add_subdirectory (${dir})
  endif ()
endforeach ()
