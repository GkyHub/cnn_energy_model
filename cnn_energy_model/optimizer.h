#pragma once
#include "model.h"
#include "layer.h"
#include <vector>
#include <functional>
#include <string>

class Optimizer {
public:
	Net _net;

public:
	~Optimizer();

	// load a network from file
	void LoadNetFromFile(const std::string fn);

	// optimize the schedule of a single layer to minimize energy
	// the optimized energy is returned
	EnergyModel OptSingleLayer(Accelerator *acc, Layer *l, bool input_ready, bool weight_ready);

	// optimize the network with each layer considered independently
	EnergyModel OptNetworkSingle(Accelerator *acc);

	// optimize over a network consider cross layer schedule
	EnergyModel OptNetworkCrossLayer(Accelerator *acc, bool *weight_ready);

	// optimize the schedule by set weights fixed in cache
	EnergyModel OptNetworkFixedWeights(Accelerator *acc);

	EnergyModel OptNetworkFixedWeightsSub(Accelerator *acc, int l, bool *weight_ready);

	// optimize the accelerator

	// calculate the energy for data read from cache 
	// and result write to cache
	static EnergyModel GetOnChipEnergy(Accelerator *acc, Layer *l);

	// get the time for calculation of a certain layer
	static double GetCalcTime(Accelerator *acc, Layer *l);

	// Integer variable minizer by direct search
	static double IntMinimizer(int min, int max, 
		int &min_var, std::function<double(int)> func);

	// Calculate the energy efficiency
	double EnergyEfficiency(EnergyModel ene);

private:
	EnergyModel _optSingleLayer(Accelerator *acc, Layer *l, bool input_ready, bool weight_ready);
};
