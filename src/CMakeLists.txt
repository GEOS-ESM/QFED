esma_add_subdirectory (Shared)

file (GLOB qfed_files qfed/*.py)
install (
  FILES ${qfed_files} 
  DESTINATION lib/Python/qfed
  )

install (
  PROGRAMS qfed_l3a.py qfed_l3b.py 
  DESTINATION bin
  )

install (
  FILES config.yaml pix_size_viirs_iband.txt
  DESTINATION etc
  )
