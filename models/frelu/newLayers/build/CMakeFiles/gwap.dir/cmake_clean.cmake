file(REMOVE_RECURSE
  "libgwap.pdb"
  "libgwap.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/gwap.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
