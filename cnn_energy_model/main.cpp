#include "optimizer.h"
#include "device_param.h"
#include <fstream>

Accelerator InitializeAccelerator(int i, int j, int k, bool use_rram);

int main() {

	std::ofstream csv_file[5];
	csv_file[0].open("ss_vgg11_conv.csv", std::ios::out);
	csv_file[1].open("ss_vgg11_conv_f16.csv", std::ios::out);
	csv_file[2].open("ss_vgg11_conv_f32.csv", std::ios::out);
	csv_file[3].open("ss_vgg11_conv_f64.csv", std::ios::out);
	csv_file[4].open("ss_vgg11_conv_f128.csv", std::ios::out);

	Optimizer opt;
	opt.LoadNetFromFile("vgg-11-conv.txt");
	std::cout << "load completed!" << std::endl;

	for (int k = 0; k < 5; k++) {
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				std::cout << "optimization on (" << i << "," << j << ")" << std::endl;

				Accelerator acc = InitializeAccelerator(i, j, k, false);

				EnergyModel ene1 = opt.OptNetworkSingle(&acc);
				ene1.PrintCSV(csv_file[k]);
				csv_file[k] << " ,";
				//std::cout << "Single Layer Optimization:" << std::endl;
				//std::cout << ene1 << std::endl;

				bool *weight_ready = new bool[opt._net.size()];
				for (int i = 0; i < opt._net.size(); i++) {
					weight_ready[i] = false;
				}
				EnergyModel ene2 = opt.OptNetworkCrossLayer(&acc, weight_ready);
				ene2.PrintCSV(csv_file[k]);
				csv_file[k] << " ,";
				//std::cout << "Cross Layer Optimization:" << std::endl;
				//std::cout << ene2 << std::endl;

				EnergyModel ene3 = opt.OptNetworkFixedWeights(&acc);
				ene3.PrintCSV(csv_file[k]);
				csv_file[k] << std::endl;
				//std::cout << "Fixed Weights Optimization:" << std::endl;
				//std::cout << ene3 << std::endl;
			}
			csv_file[k] << std::endl;
		}
	}

	csv_file[0].close();
	csv_file[1].close();
	csv_file[2].close();
	csv_file[3].close();
	csv_file[4].close();	

	return 0;
}

Accelerator InitializeAccelerator(int i, int j, int k, bool use_rram)
{
	Accelerator acc;
	// DDR configuration
	acc._ddr._rd_bw = DDR_BW;
	acc._ddr._wr_bw = DDR_BW;
	acc._ddr._unit_rd_ene = DDR_RD_ENE_PER_BYTE;
	acc._ddr._unit_wr_ene = DDR_WR_ENE_PER_BYTE;
	acc._ddr._bg_pwr = DDR_BG_PWR;
	// iobuffer configuration
	acc._iobuf._size = SRAM_UNIT_SIZE[i] * PIXEL_P;
	acc._iobuf._rd_bw = SRAM_UNIT_RD_BW[i] * PIXEL_P;
	acc._iobuf._wr_bw = SRAM_UNIT_WR_BW[i] * PIXEL_P;
	acc._iobuf._unit_rd_ene = SRAM_UNIT_RD_ENE[i];
	acc._iobuf._unit_wr_ene = SRAM_UNIT_WR_ENE[i];
	acc._iobuf._bg_pwr = SRAM_UNIT_BG_PWR[i] * PIXEL_P * 2;
	// weight buffer configuration
	if (use_rram) {
		acc._weight._size = RRAM_UNIT_SIZE[j] * PIXEL_P;
		acc._weight._rd_bw = RRAM_UNIT_RD_BW[j] * PIXEL_P;
		acc._weight._wr_bw = RRAM_UNIT_WR_BW[j] * PIXEL_P;
		acc._weight._unit_rd_ene = RRAM_UNIT_RD_ENE[j];
		acc._weight._unit_wr_ene = RRAM_UNIT_WR_ENE[j];
		acc._weight._bg_pwr = RRAM_UNIT_BG_PWR[j] * PIXEL_P;
	}
	else {
		acc._weight._size = SRAM_UNIT_SIZE[j] * PIXEL_P;
		acc._weight._rd_bw = SRAM_UNIT_RD_BW[j] * PIXEL_P;
		acc._weight._wr_bw = SRAM_UNIT_WR_BW[j] * PIXEL_P;
		acc._weight._unit_rd_ene = SRAM_UNIT_RD_ENE[j];
		acc._weight._unit_wr_ene = SRAM_UNIT_WR_ENE[j];
		acc._weight._bg_pwr = SRAM_UNIT_BG_PWR[j] * PIXEL_P;
	}
	// mac array configuration
	acc._input_map_p = CHANNEL_P;
	acc._output_map_p = CHANNEL_P;
	acc._pixel_p = PIXEL_P;
	acc._mac_ene = MAC_ENE;
	acc._mac_freq = MAC_FREQ;

	// mac buffer configuration
	acc._acc_buf._size = FIFO_SIZE[k];
	acc._acc_buf._unit_rd_ene = FIFO_UNIT_RD_ENE[k];
	acc._acc_buf._unit_wr_ene = FIFO_UNIT_WR_ENE[k];

	return acc;
}