#
# Cmake rules for creating python modules with f2py and 
#

# The OVERRIDE tells esma_set_this() to name the library as
# what we set here rather than the name of the directory

esma_set_this ( OVERRIDE plumerise )

# cmake requirements
# ------------------
  find_package(F2PY3 REQUIRED)


# Libray
# ------
  esma_add_library (${this}
    SRCS            LockePlume_Mod.F90
                    rconstants.F90
                    FreitasPlume_Mod.F90
                    qsat_Mod.F90
  )

# Module: FreitasPlume
# --------------------
  esma_add_f2py3_module(FreitasPlume_
     SOURCES        FreitasPlume_py.F90
     DESTINATION    lib/Python/plumerise
     LIBRARIES      plumerise
     INCLUDEDIRS    ${CMAKE_CURRENT_BINARY_DIR}
                    ${CMAKE_BINARY_DIR}/lib
                    ${include_${this}}
   )
   add_dependencies(FreitasPlume_ ${this})

# Module: Locke
# -------------
  esma_add_f2py3_module(LockePlume_
     SOURCES        LockePlume_py.F90
     DESTINATION    lib/Python/plumerise
     LIBRARIES      plumerise
     INCLUDEDIRS    ${CMAKE_CURRENT_BINARY_DIR}
                    ${CMAKE_BINARY_DIR}/lib
                    ${include_${this}}
   )
   add_dependencies(LockePlume_ ${this})

   
