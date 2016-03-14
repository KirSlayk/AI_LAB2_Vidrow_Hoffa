/*
NOT_x3 * x4 * (x1 + x2)

x1	x2	x3	x4	F
0	0	0	0	0
0	0	0	1	0
0	0	1	0	0
0	0	1	1	0
0	1	0	0	0
0	1	0	1	1
0	1	1	0	0
0	1	1	1	0
1	0	0	0	0
1	0	0	1	1
1	0	1	0	0
1	0	1	1	0
1	1	0	0	0
1	1	0	1	1
1	1	1	0	0
1	1	1	1	0

(1/2 * (net/(1+abs(net)) + 1))' = 1/(2*(abs(net) + 1)^2)
*/
#include <iostream>
#include <math.h>
#include <fstream>

using namespace std;

float netFunc(bool **xArr, size_t n, size_t col, float *wArr){
	float net = 0;
	for (size_t i = 0; i < n; ++i){
		net += static_cast<int>(xArr[i][col]) * wArr[i];
	}
	return net += wArr[n];
}

bool outFunc(float net){
	return net >= 0 ? true : false;
}

void getTruthVectorOfValueForMySimpleFunc(size_t *vectorTruthFunction, bool **xArr, size_t tableTruthNumOfCol){
	for (size_t i = 0; i < tableTruthNumOfCol; ++i){
		vectorTruthFunction[i] = ((xArr[0][i] | xArr[1][i]) & (!xArr[2][i]) & xArr[3][i]) == true ? 1 : 0;
	}
}


void calcValueOfEraError(size_t *vectorCalcFunction, size_t *vectorTruthFunction, size_t tableTruthNumOfCol, int &squereError){
	squereError = 0;
	for (size_t i = 0; i < tableTruthNumOfCol; ++i){
		squereError += vectorCalcFunction[i] ^ vectorTruthFunction[i];
	}
}

void calcNewValuesOfWARR(float *wArr, float *wDeltaArr, bool **xArr, size_t i, size_t tableTruthNumOfElement, float etto, int delta, float net, bool isSigmoid){
	for (size_t j = 0; j < tableTruthNumOfElement; j++){
		wArr[j] = wArr[j] + wDeltaArr[j];
		if (!isSigmoid)
			wDeltaArr[j] = etto * delta * static_cast<int>(xArr[j][i]);

		else wDeltaArr[j] = static_cast<float>(etto * delta * static_cast<int>(xArr[j][i])) / (2 * static_cast<float>(powf(abs(net) + 1, 2)));
	}
	wArr[tableTruthNumOfElement] = wArr[tableTruthNumOfElement] + wDeltaArr[tableTruthNumOfElement];
	if (!isSigmoid)
		wDeltaArr[tableTruthNumOfElement] = etto * delta;
	else wDeltaArr[tableTruthNumOfElement] = static_cast<float>(etto * delta) / (2 * static_cast<float>(powf(abs(net) + 1, 2)));

}

void printAnswer(size_t *vectorCalcFunction, size_t tableTruthNumOfCol, float *wArr, size_t tableTruthNumOfElement, int squereError, ofstream &out, size_t coutnEra){
	if (out.is_open()){
		out << "Номер эпохи " << coutnEra << "\t";
		out << "Y = ( ";
		for (size_t i = 0; i < tableTruthNumOfCol; ++i){
			out << vectorCalcFunction[i] << " ";
		}
		out << "),\n";
		out << "W = ( ";
		for (size_t i = 0; i < tableTruthNumOfElement + 1; ++i){
			out << wArr[i] << " ";
		}
		out << "),\t";
		out << "E = " << squereError << "\n";
	}
}

void main(int argc, char*argv[]){
	setlocale(LC_ALL, "rus");

	size_t tableTruthNumOfElement = 4,
		tableTruthNumOfCol = static_cast<size_t>(pow(2, tableTruthNumOfElement));

	bool **xArr = new bool*[tableTruthNumOfElement];

	for (size_t i = 0; i < tableTruthNumOfElement; ++i){
		xArr[i] = new bool[tableTruthNumOfCol];
	}

	int divider = 0;
	bool val = true;
	size_t j = 0;

	// забиваю матрицу с 4 строками x1x2x3x4 и 16 столбцами 
	for (size_t i = 0; i < tableTruthNumOfElement; ++i){
		for (j = 0, divider = tableTruthNumOfCol / (static_cast<int>(pow(2, (i + 1)))), val = true; j < tableTruthNumOfCol; ++j){
			if (!((j) % divider))
				val = !val;
			xArr[i][j] = val;
		}
	}

	size_t *vectorTruthFunction = new size_t[tableTruthNumOfCol]; // вектор истинных значений функции, которые мы в итоге должны получить
	getTruthVectorOfValueForMySimpleFunc(vectorTruthFunction, xArr, tableTruthNumOfCol); // заношу истинные значения в вектор значений функции


	float *wArr = (float*)calloc(tableTruthNumOfElement + 1, sizeof(float)); // 00000 веса связей
	float etto = 0.3; // норма обучения
	float *wDeltaArr = (float*)calloc(tableTruthNumOfElement + 1, sizeof(float)); // вектор текущих дельта значений 00000

	size_t *vectorCalcFunction = new size_t[tableTruthNumOfCol]; // вектор значений функций для обучения

	int delta = 0; // текущая дельта
	int squereError = -1; // ошибка эпохи

	ofstream out;

	if (argc > 1)
		out.open(argv[1], ios::out | ios::trunc);
	else out.open("myout.txt");

	size_t countEra = 0;

	while (squereError){
		for (size_t i = 0; i < tableTruthNumOfCol; ++i){
			float net = netFunc(xArr, tableTruthNumOfElement, i, wArr); // считаю net
			vectorCalcFunction[i] = outFunc(net); // процесс обучения
			delta = vectorTruthFunction[i] - vectorCalcFunction[i]; // считаю дельту
			// 1 2 3 5 7 9 10 11 12 9ть обучающих векторов для сигмоидальной функции
			calcNewValuesOfWARR(wArr, wDeltaArr, xArr, i, tableTruthNumOfElement, etto, delta, net, true); // пересчитываю вектор весов
		}
		calcValueOfEraError(vectorCalcFunction, vectorTruthFunction, tableTruthNumOfCol, squereError); // высчитываю ошибку в этой эре
		if (out.is_open())
			printAnswer(vectorCalcFunction, tableTruthNumOfCol, wArr, tableTruthNumOfElement, squereError, out, countEra++);
	}

	getchar();

}