#include "optimizer.h"
#include <iostream>
#include <climits>
#include <fstream>

#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define CEIL_DIV(X, Y) (((X) + (Y) - 1) / (Y))
#define ALIGN(X, BASE) ((BASE) * (CEIL_DIV((X), (BASE))))

Optimizer::~Optimizer()
{
	if (!_net.empty()) {
		for each (Layer *l in _net) {
			delete l;
		}
	}
	_net.clear();
}

void Optimizer::LoadNetFromFile(const std::string fn) {
	if (!_net.empty()) {
		for each (Layer *l in _net) {
			delete l;
		}
	}
	_net.clear();

	int layer_num;
	std::ifstream is(fn, std::ios::in);
	is >> layer_num;
	_net.resize(layer_num);

	for (int i = 0; i < layer_num; i++) {
		Layer *l = new Layer;
		is >> (*l);
		_net[i] = l;
	}
	return;
}

// optimize the schedule of a single layer to minimize energy
// the optimized energy is returned
EnergyModel Optimizer::_optSingleLayer(Accelerator *acc, Layer *l, bool input_ready, bool weight_ready)
{
	EnergyModel ene;

	int input_map_size = l->GetInputMapSize();
	int output_map_size = l->GetOutputMapSize();
	int weight_size = l->GetWeightSize();
	
	bool cut_output = output_map_size > acc->_iobuf._size;

	// get the necessary energy for data read from cache
	// and result write to cache
	ene = GetOnChipEnergy(acc, l);
	double calc_time = GetCalcTime(acc, l);

	// in any case choose the data reuse pattern.
	// Considering the buffer bandwidth limitation, reuse slow buffer
	// may reduce the background power
	// otherwise, cutting param should be chosen to optimize the energy
	double output_trans_time = cut_output ? l->GetOutputMapSize() / acc->WriteMapBw() : 0.0;
	int input_trans_size;
	int weight_trans_size;

	// case 1: calculate pixel first, reuse weights
	// then, each feature map will be loaded multiple times
	// weights are loaded once
	EnergyModel case1_ene;
	int cut_channel = CEIL_DIV(weight_size, acc->_weight._size);
	input_trans_size = input_ready ? 0 : (input_map_size * cut_channel);
	weight_trans_size = weight_ready ? 0 : weight_size;

	case1_ene._rd_ddr = acc->_ddr._unit_rd_ene * (input_trans_size + weight_trans_size);
	case1_ene._wr_iobuf = input_trans_size * acc->_iobuf._unit_wr_ene;
	case1_ene._wr_weight = weight_trans_size * acc->_weight._unit_wr_ene;

	double case1_trans_time = output_trans_time + 
		input_trans_size / acc->ReadMapBw() +
		weight_trans_size / acc->ReadWeightBw();
	case1_ene._bg = acc->BackgroundPower() * MAX(case1_trans_time, calc_time) * 1000;

	// case 2: calculate channel first, reuse feature map
	// then, each weight will be loaded multiple times
	// input are loaded once
	EnergyModel case2_ene;
	int cut_map = CEIL_DIV(input_map_size, acc->_iobuf._size);
	input_trans_size = input_ready ? 0 : input_map_size;
	weight_trans_size = weight_ready ? 0 : (weight_size * cut_map);

	case2_ene._rd_ddr = acc->_ddr._unit_rd_ene * (input_trans_size + weight_trans_size);
	case2_ene._wr_iobuf = input_trans_size * acc->_iobuf._unit_wr_ene;
	case2_ene._wr_weight = weight_trans_size * acc->_weight._unit_wr_ene;

	double case2_trans_time = output_trans_time + 
		input_map_size / acc->ReadMapBw() +
		weight_size * cut_map / acc->ReadWeightBw();
	case2_ene._bg = acc->BackgroundPower() * MAX(case2_trans_time, calc_time) * 1000;

	if (case1_ene.Total() < case2_ene.Total()) {
		ene = ene + case1_ene;
		//std::cout << "Chose case 1, cut channel = " << cut_channel << std::endl;
		//std::cout << "data trans time (us):  " << case1_trans_time << std::endl;
		//std::cout << "calculation time (us): " << calc_time << std::endl;
	}
	else {
		ene = ene + case2_ene;
		//std::cout << "Chose case 2, cut map = " << cut_map << std::endl;
		//std::cout << "data trans time (us):  " << case1_trans_time << std::endl;
		//std::cout << "calculation time (us): " << calc_time << std::endl;
	}

	if (cut_output) {
		ene._rd_iobuf += l->GetOutputMapSize() *	acc->_iobuf._unit_rd_ene;
		ene._wr_ddr += l->GetOutputMapSize() * acc->_ddr._unit_wr_ene;
	}

	return ene;
}

EnergyModel Optimizer::OptSingleLayer(Accelerator *acc, Layer *l, bool input_ready, bool weight_ready)
{
	EnergyModel ene;
	Layer *ker_layer = new Layer(*l);
	ker_layer->_input_map_num /= l->_group;
	ker_layer->_output_map_num /= l->_group;
	ene = _optSingleLayer(acc, ker_layer, input_ready, weight_ready);
	ene = ene * l->_group;
	delete ker_layer;
	return ene;
}

EnergyModel Optimizer::OptNetworkSingle(Accelerator *acc)
{
	EnergyModel tol_ene, cur_ene;
	for (int i = 0; i < _net.size(); i++) {
		//std::cout << "Layer " << i << std::endl;
		bool input_ready = (i > 0) && (_net[i]->GetInputMapSize() < acc->_iobuf._size);
		cur_ene = OptSingleLayer(acc, _net[i], input_ready, false);		
		//std::cout << cur_ene << std::endl;
		tol_ene = tol_ene + cur_ene;
	}
	// write result to ddr finally
	tol_ene._rd_iobuf += _net[_net.size() - 1]->GetOutputMapSize() * acc->_iobuf._unit_rd_ene;
	tol_ene._wr_ddr += _net[_net.size() - 1]->GetOutputMapSize() * acc->_ddr._unit_wr_ene;
	return tol_ene;
}

// optimize over a network consider cross layer schedule
EnergyModel Optimizer::OptNetworkCrossLayer(Accelerator *acc, bool *weight_ready)
{
	int layer_num = _net.size();
	EnergyModel *opt_ene = new EnergyModel[layer_num];
	EnergyModel *on_chip_ene = new EnergyModel[layer_num];
	double *calc_time = new double[layer_num];
	bool *fits_in_buf = new bool[layer_num];
	// cut[i]: for layer 0-i, the last group of layer is
	// cut[i]~j
	int *cut = new int[layer_num];
	bool *input_ready = new bool[layer_num + 1];

	// calculate the necessary on-chip energy for all the layers first
	for (int i = 0; i < layer_num; i++) {
		on_chip_ene[i] = GetOnChipEnergy(acc, _net[i]);
		calc_time[i] = GetCalcTime(acc, _net[i]);
		fits_in_buf[i] = _net[i]->GetInputMapSize() < acc->_iobuf._size;
	}

	// initialize the first layer
	opt_ene[0] = OptSingleLayer(acc, _net[0], false, weight_ready[0]);
	cut[0] = 0;
	input_ready[0] = false;
	input_ready[1] = _net[0]->GetOutputMapSize() < acc->_iobuf._size;
	
	for (int i = 1; i < layer_num; i++) {
		// first try no merge
		opt_ene[i] = OptSingleLayer(acc, _net[i], input_ready[i], weight_ready[i]) + opt_ene[i-1];
		cut[i] = i;
		input_ready[i + 1] = _net[i]->GetOutputMapSize() < acc->_iobuf._size;
		int tol_weight_size = (!weight_ready[i]) ? _net[i]->GetWeightSize() : 0;

		// try to merge layer j to i
		EnergyModel merge_calc_ene = on_chip_ene[i];
		double merge_calc_time = calc_time[i];
		double merge_data_trans_time = _net[i]->GetOutputMapSize() / acc->WriteMapBw();
		merge_data_trans_time += tol_weight_size / acc->ReadMapBw();

		bool write_output = (i == (layer_num - 1)) || (!fits_in_buf[i + 1]);
		for (int j = i - 1; j >= 0; j--) {
			// merge layer j to layer i

			// if the total size of weight exceeds the size of
			// weights buffer, then we do not try to merge them
			tol_weight_size += (!weight_ready[j]) ? _net[j]->GetWeightSize() : 0;		
			if (tol_weight_size > acc->_weight._size) {
				break;
			}

			// otherwise, we check if the feature map fits into the buffer
			// right now, we assume it is okay

			// calculate the energy for the merged layer
			EnergyModel cur_ene;
			if (j > 0) {
				// if this is not the first layer, first add all the prev energy
				cur_ene = cur_ene + opt_ene[j - 1];
			}
			
			// then add the feature map input energy if needed
			if (!input_ready[j]) {
				cur_ene._rd_ddr += _net[j]->GetInputMapSize() *
					acc->_ddr._unit_rd_ene;
				cur_ene._wr_iobuf += _net[j]->GetInputMapSize() *
					acc->_iobuf._unit_rd_ene;	
			}

			// add necessary on-chip energy
			merge_calc_ene = merge_calc_ene + on_chip_ene[j];
			cur_ene = cur_ene + merge_calc_ene;

			// add weight transfer energy
			cur_ene._rd_ddr += tol_weight_size * acc->_ddr._unit_rd_ene;
			cur_ene._wr_weight += tol_weight_size * acc->_weight._unit_wr_ene;

			// add background energy
			merge_data_trans_time += tol_weight_size / acc->ReadWeightBw();
			merge_calc_time += calc_time[j];
			double data_trans_time = merge_data_trans_time + ((!input_ready[j]) ? 
				(_net[j]->GetInputMapSize() / acc->ReadMapBw()) : 0);
			double time = MAX(merge_calc_time, data_trans_time);
			cur_ene._bg += time * acc->BackgroundPower() * 1000;

			write_output = write_output || (!fits_in_buf[j + 1]);

			// judge if this is a better choice
			if (cur_ene.Total() < opt_ene[i].Total()) {
				cut[i] = j;
				opt_ene[i] = cur_ene;
				input_ready[i + 1] = !write_output;
			}
		}
	}

	/*
	for (int i = 0; i < layer_num; i++) {
		std::cout << "Layer " << i << ": " << cut[i] << "\t" << input_ready[i] << std::endl;
		std::cout << opt_ene[i] << std::endl;
	}*/

	// write the final result back to ddr
	EnergyModel res = opt_ene[layer_num - 1];
	res._rd_iobuf += _net[layer_num - 1]->GetOutputMapSize() * acc->_iobuf._unit_rd_ene;
	res._wr_ddr += _net[layer_num - 1]->GetOutputMapSize() * acc->_ddr._unit_wr_ene;

	delete[] opt_ene;
	delete[] cut;
	delete[] input_ready;
	delete[] on_chip_ene;
	delete[] calc_time;
	delete[] fits_in_buf;

	return res;
}

// optimize the schedule by set weights fixed in cache
EnergyModel Optimizer::OptNetworkFixedWeights(Accelerator *acc)
{
	int tol_weight_size = 0;
	int layer_num = _net.size();

	for each (Layer *l in _net) {
		tol_weight_size += l->GetWeightSize();
	}

	bool *weight_ready = new bool[layer_num];

	// if all the weights fits in the cache, then fits them in
	if (tol_weight_size <= acc->_weight._size) {
		std::cout << "Put all the weights in cache." << std::endl;
		for (int i = 0; i < layer_num; i++) {
			weight_ready[i] = true;
		}
		return OptNetworkCrossLayer(acc, weight_ready);
	}

	// otherwise, search a result in a recursion mode
	EnergyModel res = OptNetworkFixedWeightsSub(acc, 0, weight_ready);
	for (int i = 0; i < layer_num; i++) {
		//std::cout << weight_ready[i] << std::endl;
	}
	delete[] weight_ready;
	return res;
}

EnergyModel Optimizer::OptNetworkFixedWeightsSub(Accelerator *acc, int l, bool *weight_ready)
{
	if (l >= (_net.size() - 1)) {
		weight_ready[_net.size() - 1] = false;
		return OptNetworkCrossLayer(acc, weight_ready);
	}

	// if the rest weights are larger than the RAM
	// then skip this layer
	if (_net[l]->GetWeightSize() > acc->_weight._size) {
		weight_ready[l] = false;
		return OptNetworkFixedWeightsSub(acc, l + 1, weight_ready);
	}

	// compare if set the weights of this layer in the RAM
	EnergyModel ene1;
	EnergyModel ene2;
	EnergyModel ene;

	bool *weight_ready1 = new bool[_net.size()];
	bool *weight_ready2 = new bool[_net.size()];
	memcpy(weight_ready1, weight_ready, _net.size());
	memcpy(weight_ready2, weight_ready, _net.size());
	weight_ready1[l] = false;
	weight_ready2[l] = true;

	Accelerator acc2 = *acc;
	acc2._weight._size -= _net[l]->GetWeightSize();

	ene1 = OptNetworkFixedWeightsSub(acc, l + 1, weight_ready1);
	ene2 = OptNetworkFixedWeightsSub(&acc2, l + 1, weight_ready2);

	if (ene1.Total() < ene2.Total()) {
		memcpy(weight_ready, weight_ready1, _net.size());
		ene = ene1;
	}
	else {
		memcpy(weight_ready, weight_ready2, _net.size());
		ene = ene2;
	}

	delete[] weight_ready1;
	delete[] weight_ready2;
	return ene;
}


// calculate the energy for data read from cache 
// and result write to cache
EnergyModel Optimizer::GetOnChipEnergy(Accelerator *acc, Layer *l)
{
	EnergyModel ene;
	// get energy for data read from input buffer
	double input_energy_pJ;
	if (l->_kernel_str == 1 && acc->_acc_buf._size == 1) {
		ene._rd_iobuf =
			l->_input_map_y * CEIL_DIV(l->_input_map_x, acc->_pixel_p) *// output pixel group number
			(l->_kernel_x + acc->_pixel_p - 1) * l->_kernel_y *			// input pixel group size
			acc->_iobuf._unit_rd_ene;									// unit energy
	}
	else {
		ene._rd_iobuf = (l->_input_map_x / l->_kernel_str) *
			(l->_input_map_y / l->_kernel_str) *
			(l->_kernel_x * l->_kernel_y) *
			acc->_iobuf._unit_rd_ene;
	}	

	// get energy for result write to output buffer
	ene._wr_iobuf =
		l->GetOutputMapSize() * acc->_iobuf._unit_wr_ene;

	// get energy for weight read from weight buffer
	int output_map_x = l->_input_map_x / l->_kernel_str;
	int output_map_y = l->_input_map_y / l->_kernel_str;
	ene._rd_weight = 
		l->GetWeightSize() * 
		CEIL_DIV(output_map_x * CEIL_DIV(output_map_y, acc->_pixel_p), acc->_acc_buf._size) *
		acc->_weight._unit_rd_ene;

	// get calculation energy
	ene._calc = l->GetMacNum() * acc->_mac_ene;
	ene._calc += (output_map_x * output_map_y * l->_output_map_num) *
		(CEIL_DIV(l->_input_map_num, acc->_input_map_p) * l->_kernel_x * l->_kernel_y - 1) *
		(acc->_acc_buf._unit_rd_ene + acc->_acc_buf._unit_wr_ene);

	return ene;
}

double Optimizer::GetCalcTime(Accelerator *acc, Layer *l)
{
	double cycle_num;
	cycle_num = (l->_input_map_y / l->_kernel_str) *
		CEIL_DIV(l->_input_map_x / l->_kernel_str, acc->_pixel_p);
	cycle_num *= CEIL_DIV(l->_input_map_num, acc->_input_map_p) *
		CEIL_DIV(l->_output_map_num, acc->_output_map_p);
	cycle_num *= l->_kernel_x * l->_kernel_y;

	return cycle_num / acc->_mac_freq;
}

// Integer variable minizer by direct search
double Optimizer::IntMinimizer(int min, int max, int &min_var,
	std::function<double(int)> func)
{
	double val;
	double min_val = func(min);
	min_var = min;
	for (int i = min + 1; i <= max; i++) {
		if ((val = func(i)) < min_val) {
			min_var = i;
			min_val = val;
		}
	}
	return min_val;
}

double Optimizer::EnergyEfficiency(EnergyModel ene)
{
	double mac_num = 0;
	for each (Layer *l in _net) {
		mac_num += l->GetMacNum();
	}
	return ene.Total() / mac_num;
}