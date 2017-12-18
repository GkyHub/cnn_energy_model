#pragma once
#include <vector>
#include <iostream>
#include <string>

class Layer {
public:
	// input feature map size
	int _input_map_x;
	int _input_map_y;
	// convolution kernel size
	int _kernel_x;
	int _kernel_y;
	int _kernel_str;
	// channel number;
	int _input_map_num;
	int _output_map_num;
	// is pooling
	bool _is_pooling;
	// pool size
	int _pool_x;
	int _pool_y;
	int _pool_str;

	int _group;

public:
	void GetOutputMapShape(int &output_map_x, int &output_map_y);

	int GetInputMapSize();

	int GetOutputMapSize();

	int GetWeightSize();

	double GetMacNum();

	friend inline std::istream &operator >> (std::istream &is, Layer &l)
	{
		// input map size
		is >> l._input_map_x;
		l._input_map_y = l._input_map_x;
		// channel number
		is >> l._input_map_num >> l._output_map_num;
		// convolution kernel size and stride
		is >> l._kernel_x;
		l._kernel_y = l._kernel_x;
		is >> l._kernel_str;
		// group
		is >> l._group;
		// pooling parameter
		is >> l._is_pooling;
		if (l._is_pooling) {
			is >> l._pool_x;
			l._pool_y = l._pool_x;
			is >> l._pool_str;
		}
		return is;
	}
}; 

typedef std::vector<Layer *> Net;
typedef Net::iterator pNet;
