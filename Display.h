#pragma once
#include "Parameters.h"

#include "GL/glew.h"
#include "GL/freeglut.h"
#include <cuda_gl_interop.h>

#include "helper_functions.h" // StopWatchInterface

typedef unsigned char byte;

// External pointers and function to update the bitmap
//extern byte*				SLM_IMAGE_PTR;


// Declare display related functions
void init_window(const Parameters&);

// It is not const because it is internally copied
// it won't be altered though
void display_phasemap(byte* image_ptr);
