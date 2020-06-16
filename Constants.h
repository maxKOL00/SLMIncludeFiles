#pragma once
#define USE_GUI true
constexpr unsigned int SLM_PX_X = 1920;
constexpr unsigned int SLM_PX_Y = 1152;

constexpr unsigned int NUMBER_OF_PIXELS_UNPADDED = (SLM_PX_X > SLM_PX_Y ? SLM_PX_Y: SLM_PX_X);

constexpr unsigned int BLOCK_SIZE = 256;
constexpr unsigned int NUMBER_OF_PIXELS_PADDED = 8192;

constexpr unsigned int NUM_BLOCKS_PADDED = (NUMBER_OF_PIXELS_PADDED * NUMBER_OF_PIXELS_PADDED + BLOCK_SIZE - 1) / BLOCK_SIZE;
constexpr unsigned int NUM_BLOCKS_UNPADDED = (NUMBER_OF_PIXELS_UNPADDED * NUMBER_OF_PIXELS_UNPADDED + BLOCK_SIZE - 1) / BLOCK_SIZE;
constexpr unsigned int NUM_BLOCKS_SLM = (SLM_PX_Y * SLM_PX_X + BLOCK_SIZE - 1) / BLOCK_SIZE;
