#pragma once
//=============================================================================
// System Configuration
//=============================================================================
const int MAC_FREQ = 1000;	// 1GHz
const int CHANNEL_P = 8;	// input and output parallelism
const int PIXEL_P = 8;		// pixel parallelism
const double MAC_ENE = 0.07021;	// estimated by scaling down from 45nm 32bit operations

//=============================================================================
// DDR device parameter
//=============================================================================

const double DDR_FREQ_MHZ = 800;	// working frequency, both pos and neg edge
const int    DDR_CHIP_NUM = 2;		// number of chips
const double DDR_CHIP_BW = 16;		// chip bit width

								// single chip parameter from calculator
const double DDR_BG_PWR_PER_CHIP_MW = 52.8;			// background power
const double DDR_RD_PWR_RATIO = 1.8139 + 1.1524;	// power(mW) / (1% bandwidth usage)
const double DDR_WR_PWR_RATIO = 1.7646 + 0.6487;	// power(mW) / (1% bandwidth usage)
const double DDR_ACT_PWR_RATIO = 0.2337;			// power(mW) / (1% bandwidth usage)

// DDR bandwidth										
const double DDR_BW = DDR_FREQ_MHZ * 2 * DDR_CHIP_BW / 8 * DDR_CHIP_NUM;

// I/O energy
const double DDR_RD_ENE_PER_BYTE =
	(DDR_ACT_PWR_RATIO + DDR_RD_PWR_RATIO) /
	(DDR_FREQ_MHZ * 2 * DDR_CHIP_BW / 8 * 0.01) * 1000;
const double DDR_WR_ENE_PER_BYTE =
	(DDR_ACT_PWR_RATIO + DDR_WR_PWR_RATIO) /
	(DDR_FREQ_MHZ * 2 * DDR_CHIP_BW / 8 * 0.01) * 1000;

// background power
const double DDR_BG_PWR = DDR_BG_PWR_PER_CHIP_MW * DDR_CHIP_NUM;

//=============================================================================
// SRAM device parameter
//=============================================================================

const int SRAM_UNIT_SIZE[5] = {
	16 * 1024,
	32 * 1024,
	64 * 1024,
	128 * 1024,
	256 * 1024,
};

const double SRAM_UNIT_RD_BW[5] = {
	9145, 8609, 16831, 10977, 10977
};

const double SRAM_UNIT_WR_BW[5] = {
	5147, 4973, 11763, 10451, 10451
};

const double SRAM_UNIT_RD_ENE[5] = {
	3.057 / 8,
	6.432 / 8,
	6.780 / 8,
	7.931 / 8,
	11.562 / 8
};

const double SRAM_UNIT_WR_ENE[5] = {
	0.556 / 8,
	1.315 / 8,
	3.777 / 8,
	2.792 / 8,
	6.424 / 8
};

const double SRAM_UNIT_BG_PWR[5] = {
	0.00134, 0.00270, 0.00600, 0.01153, 0.02306
};

//=============================================================================
// RRAM device parameter
//=============================================================================

const int RRAM_UNIT_SIZE[5] = {
	128 * 1024,
	256 * 1024,
	512 * 1024,
	1024 * 1024,
	2048 * 1024,
};

const double RRAM_UNIT_RD_BW[5] = {
	15169, 11693, 8005, 11056, 10306
};

const double RRAM_UNIT_WR_BW[5] = {
	1556, 1534, 1490, 1534, 1534
};

const double RRAM_UNIT_RD_ENE[5] = {
	67.690 / 32,
	75.071 / 32,
	88.483 / 32,
	133.189 / 32,
	231.750 / 32
};

const double RRAM_UNIT_WR_ENE[5] = {
	195.286 / 32,
	217.468 / 32,
	260.073 / 32,
	268.319 / 32,
	357.190 / 32
};

const double RRAM_UNIT_BG_PWR[5] = {
	0.04000, 0.04104, 0.04314, 0.05282, 0.07806
};

//=============================================================================
// accumulator fifo parameter
//=============================================================================
const double FIFO_SIZE[5] = {
	1, 16, 32, 64, 128
};

const double FIFO_UNIT_RD_ENE[5] = {
	0, 0.045, 0.056, 0.107, 0.12
};

const double FIFO_UNIT_WR_ENE[5] = {
	0, 0.022, 0.031, 0.083, 0.094
};