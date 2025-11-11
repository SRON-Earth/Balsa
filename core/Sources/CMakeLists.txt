# -------------------------------------------------------
# Generate config.h from template
# -------------------------------------------------------
configure_file(config.h.in config.h)

# -------------------------------------------------------
# Add tests
# -------------------------------------------------------
add_test(NAME testsuite COMMAND balsa_test)

# -------------------------------------------------------
# Build shared library
# -------------------------------------------------------
add_library(balsa SHARED
    fileio.cpp
    modelevaluation.cpp
    serdes.cpp
    weightedcoin.cpp
)
target_include_directories(balsa PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Build static library (now included in build)
add_library(balsa-static STATIC
    fileio.cpp
    modelevaluation.cpp
    serdes.cpp
    weightedcoin.cpp
)
set_property(TARGET balsa-static PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(balsa-static PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# -------------------------------------------------------
# Build executables and link with balsa library
# -------------------------------------------------------
macro(add_balsa_tool tool_name src_file)
    add_executable(${tool_name} ${src_file})
    target_link_libraries(${tool_name} PRIVATE balsa)
endmacro()

add_balsa_tool(balsa_train balsa_train.cpp)
add_balsa_tool(balsa_classify balsa_classify.cpp)
add_balsa_tool(balsa_merge balsa_merge.cpp)
add_balsa_tool(balsa_generate balsa_generate.cpp)
add_balsa_tool(balsa_print balsa_print.cpp)
add_balsa_tool(balsa_measure balsa_measure.cpp)
add_balsa_tool(balsa_featureimportance balsa_featureimportance.cpp)
add_balsa_tool(balsa_convert balsa_convert.cpp)
add_balsa_tool(balsa_test balsa_test.cpp)

# -------------------------------------------------------
# Installation targets
# -------------------------------------------------------

# Install the shared library
install(TARGETS balsa
    LIBRARY DESTINATION lib
)

# Install static library
install(TARGETS balsa-static
    ARCHIVE DESTINATION lib
)

# Install executables
install(TARGETS
    balsa_train
    balsa_classify
    balsa_merge
    balsa_generate
    balsa_print
    balsa_measure
    balsa_featureimportance
    balsa_convert
    balsa_test
    RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# Install config.h (generated)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/config.h
    DESTINATION include
)

