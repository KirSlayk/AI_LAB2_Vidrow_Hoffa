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
#include <algorithm>

using namespace std;

// ����������� phiFunction
float phiFunctionCalc(bool **xArr, size_t tableTruthNumOfElement, size_t col, bool *cArr){
	int degree = 0;
	for (size_t i = 0; i < tableTruthNumOfElement; ++i){
		degree += static_cast<int>(pow((static_cast<int>(xArr[i][col])) - static_cast<int>(cArr[i]), 2));
	}
	return exp(-degree);
}

// �������� net
float netFunc(float *phiArr, size_t JSize, float *vArr){
	float net = 0;
	for (size_t i = 0; i < JSize; ++i){
		net += phiArr[i] * vArr[i];
	}
	return net += vArr[JSize];
}

// Y
bool outFunc(float net){
	return net >= 0 ? true : false;
}

// �������, ������������ �������� �������� �������� �������
void getTruthVectorOfValueForMySimpleFunc(size_t *vectorTruthFunction, bool **xArr, size_t tableTruthNumOfCol){
	for (size_t i = 0; i < tableTruthNumOfCol; ++i){
		vectorTruthFunction[i] = ((xArr[0][i] | xArr[1][i]) & (!xArr[2][i]) & xArr[3][i]) == true ? 1 : 0;
	}
}

// ������� �������� ������ ���
void calcValueOfEraError(size_t *vectorCalcFunction, size_t *vectorTruthFunction, size_t tableTruthNumOfCol, int &squereError){
	squereError = 0;
	for (size_t i = 0; i < tableTruthNumOfCol; ++i){
		squereError += vectorCalcFunction[i] ^ vectorTruthFunction[i];
	}
}

// ������� �������� ������� ����� ������
void calcNewValuesOfVARR(float *vArr, float *vDeltaArr, float *phiArr, size_t JSize, int delta, float etto = 0.3){
	for (size_t j = 0; j < JSize; j++){
		vArr[j] = vArr[j] + vDeltaArr[j];
		vDeltaArr[j] = static_cast<float>(etto * delta * phiArr[j]);
	}
	vArr[JSize] = vArr[JSize] + vDeltaArr[JSize];
	vDeltaArr[JSize] = static_cast<float>(etto * delta);
}

// ������� ������
void printAnswer(size_t *vectorCalcFunction, size_t tableTruthNumOfCol, float *vArr, size_t tableTruthNumOfElement, int squereError, ofstream &out, size_t coutnEra){
	if (out.is_open()){
		out << "����� ����� " << coutnEra << "\t";
		out << "Y = ( ";
		for (size_t i = 0; i < tableTruthNumOfCol; ++i){
			out << vectorCalcFunction[i] << " ";
		}
		out << "),\n";
		out << "W = ( ";
		for (size_t i = 0; i < tableTruthNumOfElement + 1; ++i){
			out << vArr[i] << " ";
		}
		out << "),\t";
		out << "E = " << squereError << "\n";
	}
}

// �������, ������� ���������� �������������� ������� � ������ ��������� ��������
bool isTeachVector(size_t *teachVector, size_t length, size_t i){
	for (size_t j = 0; j < length; ++j)
		if (i == teachVector[j])
			return true;
	return false;
}

void main(int argc, char*argv[]){
	setlocale(LC_ALL, "rus");

	// ������� ������� ����������
	size_t tableTruthNumOfElement = 4,
		tableTruthNumOfCol = static_cast<size_t>(pow(2, tableTruthNumOfElement));

	bool **xArr = new bool*[tableTruthNumOfElement];

	for (size_t i = 0; i < tableTruthNumOfElement; ++i){
		xArr[i] = new bool[tableTruthNumOfCol];
	}

	int divider = 0;
	bool val = true;
	size_t j = 0;

	// ������� ������� � 4 �������� x1x2x3x4 � 16 ��������� 
	for (size_t i = 0; i < tableTruthNumOfElement; ++i){
		for (j = 0, divider = tableTruthNumOfCol / (static_cast<int>(pow(2, (i + 1)))), val = true; j < tableTruthNumOfCol; ++j){
			if (!((j) % divider))
				val = !val;
			xArr[i][j] = val;
		}
	}

	size_t *vectorTruthFunction = new size_t[tableTruthNumOfCol]; // ������ �������� �������� �������, ������� �� � ����� ������ ��������
	getTruthVectorOfValueForMySimpleFunc(vectorTruthFunction, xArr, tableTruthNumOfCol); // ������ �������� �������� � ������ �������� �������


	size_t sizePhiArr = 0;
	bool oneOrZero = true;
	for (size_t i = 0; i < tableTruthNumOfCol; ++i){
		sizePhiArr += vectorTruthFunction[i];
	}
	if (sizePhiArr == min(sizePhiArr, tableTruthNumOfCol - sizePhiArr))
		oneOrZero = true;
	else oneOrZero = false;

	sizePhiArr = min(sizePhiArr, tableTruthNumOfCol - sizePhiArr);
	float *phiArr = (float*)calloc(sizePhiArr, sizeof(float));

	bool **cArr = new bool*[sizePhiArr];
	for (size_t i = 0; i < sizePhiArr; ++i)
		cArr[i] = new bool[tableTruthNumOfElement];

	size_t count = 0;
	for (size_t i = 0; i < tableTruthNumOfCol; ++i){
		if (oneOrZero && vectorTruthFunction[i] == 1){
			for (size_t j = 0; j < tableTruthNumOfElement; ++j){
				cArr[count][j] = xArr[j][i];
			}
			count++;
		}
		else if (!oneOrZero && vectorTruthFunction[i] == 0){
			for (size_t j = 0; j < tableTruthNumOfElement; ++j){
				cArr[count][j] = xArr[j][i];
			}
			count++;
		}
		
	}

	float *vArr = (float*)calloc(sizePhiArr + 1, sizeof(float)); // 00000 ���� ������
	//float etto = 0.3; // ����� ��������
	float *vDeltaArr = (float*)calloc(sizePhiArr + 1, sizeof(float)); // ������ ������� ������ �������� 00000

	size_t *vectorCalcFunction = new size_t[tableTruthNumOfCol]; // ������ �������� ������� ��� ��������

	int delta = 0; // ������� ������
	int squereError = -1; // ������ �����

	ofstream out;

	if (argc > 1)
		out.open(argv[1], ios::out | ios::trunc);
	else out.open("myout.txt");

	size_t countEra = 0;

	// 1 2 3 5 7 9 10 11 12 9�� ��������� �������� ��� ������������� �������
	size_t teachVector[9] = { 1, 2, 3, 5, 7, 9, 10, 11, 12 };


	while (squereError){
		for (size_t i = 0; i < tableTruthNumOfCol; ++i){
			for (size_t j = 0; j < sizePhiArr; ++j){
				phiArr[j] = phiFunctionCalc(xArr, tableTruthNumOfElement, i, cArr[j]);
			}
			float net = netFunc(phiArr, sizePhiArr, vArr); // ������ net
			vectorCalcFunction[i] = outFunc(net); // ������� ��������
			delta = vectorTruthFunction[i] - vectorCalcFunction[i]; // ������ ������
			if (isTeachVector(teachVector, 9, i))
				calcNewValuesOfVARR(vArr, vDeltaArr, phiArr, sizePhiArr, delta); // ������������ ������ �����
		}
		calcValueOfEraError(vectorCalcFunction, vectorTruthFunction, tableTruthNumOfCol, squereError); // ���������� ������ � ���� ���
		if (out.is_open())
			printAnswer(vectorCalcFunction, tableTruthNumOfCol, vArr, tableTruthNumOfElement, squereError, out, countEra++);
	}

}