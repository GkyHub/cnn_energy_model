#include "layer.h"
#include <fstream>

void Layer::GetOutputMapShape(int &output_map_x, int &output_map_y)
{
	output_map_x = _input_map_x / _kernel_str;
	output_map_y = _input_map_y / _kernel_str;
	if (_is_pooling) {
		output_map_x = output_map_x / _pool_str;
		output_map_y = output_map_y / _pool_str;
	}
	return;
}

int Layer::GetInputMapSize()
{
	return _input_map_x * _input_map_y * _input_map_num;
}

int Layer::GetOutputMapSize()
{
	int output_map_x, output_map_y;
	GetOutputMapShape(output_map_x, output_map_y);
	return output_map_x * output_map_y * _output_map_num;
}

int Layer::GetWeightSize()
{
	return _kernel_x * _kernel_y * _input_map_num * _output_map_num;
}

double Layer::GetMacNum()
{
	double res;
	res = _input_map_x * _input_map_y / _kernel_str / _kernel_str;
	res *= _kernel_x * _kernel_y;
	res *= _input_map_num * _output_map_num;
	return res;
}