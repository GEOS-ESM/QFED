add_subdirectory (f2py)

file (GLOB pr_files ./*.py)
install (
  FILES ${pr_files} 
  DESTINATION lib/Python/plumerise
  )

