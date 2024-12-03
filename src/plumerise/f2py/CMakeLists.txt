#
# Cmake rules for creating python modules with f2py and 
#

# cmake requirements
# ------------------

find_package(OpenMP REQUIRED COMPONENTS Fortran)

# This setting is needed on macOS to ensure
# that the desired Python is found (i.e., if
# a user has installed their own Python, it
# should be favored over the system Python or
# one from Homebrew, etc.)
set (CMAKE_FIND_FRAMEWORK LAST)

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)

include(UseF2Py)

# Library: plumerise
# ------------------
add_library (plumerise
  LockePlume_Mod.F90
  rconstants.F90
  FreitasPlume_Mod.F90
  qsat_Mod.F90
)
install(TARGETS plumerise DESTINATION lib/Python/plumerise)

# Module: FreitasPlume
# --------------------
f2py_object_library(FreitasPlume_object OBJECT)
f2py_generate_module(FreitasPlume_ FreitasPlume_py.F90 OUTPUT_VARIABLE FreitasPlume_files)
python_add_library(FreitasPlume_ MODULE "${FreitasPlume_files}" WITH_SOABI)
target_link_libraries(FreitasPlume_ PRIVATE Python::NumPy)
target_link_libraries(FreitasPlume_ PRIVATE FreitasPlume_object)
target_link_libraries(FreitasPlume_ PRIVATE plumerise)
target_link_libraries(FreitasPlume_ PRIVATE OpenMP::OpenMP_Fortran)
install(TARGETS FreitasPlume_ DESTINATION lib/Python/plumerise)

# Module: Locke
# -------------

f2py_object_library(LockePlume_object OBJECT)
f2py_generate_module(LockePlume_ LockePlume_py.F90 OUTPUT_VARIABLE LockePlume_files)
python_add_library(LockePlume_ MODULE "${LockePlume_files}" WITH_SOABI)
target_link_libraries(LockePlume_ PRIVATE Python::NumPy)
target_link_libraries(LockePlume_ PRIVATE LockePlume_object)
target_link_libraries(LockePlume_ PRIVATE plumerise)
target_link_libraries(LockePlume_ PRIVATE OpenMP::OpenMP_Fortran)
install(TARGETS LockePlume_ DESTINATION lib/Python/plumerise)
