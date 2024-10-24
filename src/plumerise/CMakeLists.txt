esma_add_subdirectory (f2py)

file (GLOB pr_files ./*.py)
install (
  FILES ${qfed_files} 
  DESTINATION lib/Python/plumerise
  )

