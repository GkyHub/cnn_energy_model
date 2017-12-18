#pragma once
#include <iostream>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

class BufferModel {
public:
	int _size;			// number of datum

	// energy model
	double _unit_rd_ene;// pJ per datum
	double _unit_wr_ene;// pJ per datum
	double _bg_pwr;		// mW

	// speed model
	double _rd_bw;		// Mega datum per second
	double _wr_bw;		// Mega datum per second
};

class Accelerator {
public:
	BufferModel _iobuf;
	BufferModel _weight;
	BufferModel _ddr;

	// MAC array model
	int _input_map_p;
	int _output_map_p;
	int _pixel_p;
	double _mac_ene;
	double _mac_freq;	// MHz
	BufferModel _acc_buf;

public:
	double BackgroundPower() {
		return _iobuf._bg_pwr + _weight._bg_pwr + _ddr._bg_pwr;
	}

	double ReadWeightBw() {
		return MIN(_ddr._rd_bw, _weight._wr_bw);
	}

	double ReadMapBw() {
		return MIN(_ddr._rd_bw, _iobuf._wr_bw);
	}

	double WriteMapBw() {
		return MIN(_ddr._wr_bw, _iobuf._rd_bw);
	}
};

class EnergyModel {
public:
	double _rd_iobuf;
	double _wr_iobuf;

	double _rd_weight;
	double _wr_weight;

	double _rd_ddr;
	double _wr_ddr;

	double _bg;
	double _calc;

public:
	EnergyModel()
	{
		_rd_iobuf = 0.0;
		_wr_iobuf = 0.0;

		_rd_weight = 0.0;
		_wr_weight = 0.0;

		_rd_ddr = 0.0;
		_wr_ddr = 0.0;

		_bg = 0.0;
		_calc = 0.0;
	}

	EnergyModel operator+(EnergyModel &b)
	{
		EnergyModel c;
		c._rd_iobuf = _rd_iobuf + b._rd_iobuf;
		c._wr_iobuf = _wr_iobuf + b._wr_iobuf;

		c._rd_weight = _rd_weight + b._rd_weight;
		c._wr_weight = _wr_weight + b._wr_weight;

		c._rd_ddr = _rd_ddr + b._rd_ddr;
		c._wr_ddr = _wr_ddr + b._wr_ddr;

		c._bg = _bg + b._bg;
		c._calc = _calc + b._calc;

		return c;
	}

	EnergyModel operator*(int p)
	{
		EnergyModel c;
		c._rd_iobuf = _rd_iobuf * p;
		c._wr_iobuf = _wr_iobuf * p;

		c._rd_weight = _rd_weight * p;
		c._wr_weight = _wr_weight * p;

		c._rd_ddr = _rd_ddr * p;
		c._wr_ddr = _wr_ddr * p;

		c._bg = _bg * p;
		c._calc = _calc * p;

		return c;
	}

	double Total()
	{
		return _rd_iobuf + _wr_iobuf +
			_rd_weight + _wr_weight +
			_rd_ddr + _wr_ddr + _bg + _calc;
	}

	friend std::ostream& operator << (std::ostream &os, EnergyModel &ene)
	{
		double total = ene.Total();
		os << "-----------------------------------" << std::endl;
		os << "RAM\t" << "Read\t\t" << "Write\t\t" << std::endl;
		os << "iobuf\t" << ene._rd_iobuf << "\t" << ene._wr_iobuf <<
			"\t(" << (ene._rd_iobuf + ene._wr_iobuf) / total * 100 << "%)\t" << std::endl;
		os << "weight\t" << ene._rd_weight << "\t" << ene._wr_weight <<
			"\t(" << (ene._rd_weight + ene._wr_weight) / total * 100 << "%)\t" << std::endl;
		os << "DDR\t" << ene._rd_ddr << "\t" << ene._wr_ddr <<
			"\t(" << (ene._rd_ddr + ene._wr_ddr) / total * 100 << "%)\t" << std::endl;
		os << "background\t" << ene._bg << "\t(" << ene._bg / total * 100 << "%)\t" << std::endl;
		os << "calculate\t" << ene._calc << "\t(" << ene._calc / total * 100 << "%)\t" << std::endl;
		os << "Total:\t" << total << std::endl;
		return os;
	}

	void PrintCSV(std::ostream &os)
	{
		os << _rd_iobuf / 1e6 << ","
			<< _rd_weight / 1e6 << ","
			<< _rd_ddr / 1e6 << ","
			<< _wr_iobuf / 1e6 << ","
			<< _wr_weight / 1e6 << ","
			<< _wr_ddr / 1e6 << ","
			<< _bg / 1e6 << ","
			<< _calc / 1e6 << ","
			<< Total() / 1e6 << ",";
		return;
	}
};
