esma_add_subdirectory (Shared)
esma_add_subdirectory (plumerise)

file (GLOB qfed_files qfed/*.py)
install (
  FILES ${qfed_files} 
  DESTINATION lib/Python/qfed
  )

install (
  PROGRAMS qfed_l3a.py qfed_l3b.py 
  DESTINATION bin
  )

install(CODE "
    file(GLOB installed_pyfiles
        \"\${CMAKE_INSTALL_PREFIX}/bin/*.py\"
        \"\${CMAKE_INSTALL_PREFIX}/lib/Python/qfed/*.py\"
    )
    foreach(installed_file \${installed_pyfiles})
        file(READ \"\${installed_file}\" contents)
        string(REPLACE \"%%INSTALL_PREFIX%%\" \"\${CMAKE_INSTALL_PREFIX}\" contents \"\${contents}\")
        file(WRITE \"\${installed_file}\" \"\${contents}\")
    endforeach()
")

install (
  FILES config.yaml emission_factors.yaml qcscalingfactors.yaml
  DESTINATION etc
  )
