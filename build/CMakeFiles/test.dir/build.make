# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/robotics/Courses/CV/lab3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/robotics/Courses/CV/lab3/build

# Include any dependencies generated for this target.
include CMakeFiles/test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test.dir/flags.make

CMakeFiles/test.dir/src/match.cpp.o: CMakeFiles/test.dir/flags.make
CMakeFiles/test.dir/src/match.cpp.o: ../src/match.cpp
CMakeFiles/test.dir/src/match.cpp.o: CMakeFiles/test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robotics/Courses/CV/lab3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test.dir/src/match.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test.dir/src/match.cpp.o -MF CMakeFiles/test.dir/src/match.cpp.o.d -o CMakeFiles/test.dir/src/match.cpp.o -c /home/robotics/Courses/CV/lab3/src/match.cpp

CMakeFiles/test.dir/src/match.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test.dir/src/match.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robotics/Courses/CV/lab3/src/match.cpp > CMakeFiles/test.dir/src/match.cpp.i

CMakeFiles/test.dir/src/match.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test.dir/src/match.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robotics/Courses/CV/lab3/src/match.cpp -o CMakeFiles/test.dir/src/match.cpp.s

# Object files for target test
test_OBJECTS = \
"CMakeFiles/test.dir/src/match.cpp.o"

# External object files for target test
test_EXTERNAL_OBJECTS =

../bin/test: CMakeFiles/test.dir/src/match.cpp.o
../bin/test: CMakeFiles/test.dir/build.make
../bin/test: /usr/local/lib/libopencv_gapi.so.4.10.0
../bin/test: /usr/local/lib/libopencv_highgui.so.4.10.0
../bin/test: /usr/local/lib/libopencv_ml.so.4.10.0
../bin/test: /usr/local/lib/libopencv_objdetect.so.4.10.0
../bin/test: /usr/local/lib/libopencv_photo.so.4.10.0
../bin/test: /usr/local/lib/libopencv_stitching.so.4.10.0
../bin/test: /usr/local/lib/libopencv_video.so.4.10.0
../bin/test: /usr/local/lib/libopencv_videoio.so.4.10.0
../bin/test: /usr/local/lib/libopencv_imgcodecs.so.4.10.0
../bin/test: /usr/local/lib/libopencv_dnn.so.4.10.0
../bin/test: /usr/local/lib/libopencv_calib3d.so.4.10.0
../bin/test: /usr/local/lib/libopencv_features2d.so.4.10.0
../bin/test: /usr/local/lib/libopencv_flann.so.4.10.0
../bin/test: /usr/local/lib/libopencv_imgproc.so.4.10.0
../bin/test: /usr/local/lib/libopencv_core.so.4.10.0
../bin/test: CMakeFiles/test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/robotics/Courses/CV/lab3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test.dir/build: ../bin/test
.PHONY : CMakeFiles/test.dir/build

CMakeFiles/test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test.dir/clean

CMakeFiles/test.dir/depend:
	cd /home/robotics/Courses/CV/lab3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/robotics/Courses/CV/lab3 /home/robotics/Courses/CV/lab3 /home/robotics/Courses/CV/lab3/build /home/robotics/Courses/CV/lab3/build /home/robotics/Courses/CV/lab3/build/CMakeFiles/test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test.dir/depend

