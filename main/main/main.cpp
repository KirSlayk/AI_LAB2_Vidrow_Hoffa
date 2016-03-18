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
#include <string.h>

using namespace std;

class VectorFunction{
private:
	// ��� ��� �������� ������ ��������� �������� 
	//�� ����� ���� ������, ��� ��� ��������
	static size_t attempt;

	size_t tableTruthNumOfElement,
		tableTruthNumOfCol,
		sizePhiArr,
		minNumOfBit;
		
	// 2^16-1
	int cFromNByK,
		numWithMinNumOfBit;

	bool **xArr, // ������� ���������� X-��
		**cArr; // ������� �������� ������� RBF-��������

	bool oneOrZero; // ���� ������ � ������� �������, ����� ��� ������
	
	float *phiArr, // ������ �������� ��������� ��
		*vArr, // ������ ������������� �����
		*vDeltaArr; // ������ ������ �������� ������������� �����.  

	size_t *vectorTruthFunction, // ������ �������� �������� �������� �������
		   *vectorCalcFunction;	// ������ �������� ������� ��� ��������
	
	// �������� net
	float netFunc(){
		float net = 0;
		for (size_t i = 0; i < sizePhiArr; ++i)
			net += phiArr[i] * vArr[i];
		return net += vArr[sizePhiArr];
	}

	// Y
	bool outFunc(float net){
		return net >= 0 ? true : false;
	}

	// ������� �������� ������ ���
	void calcValueOfEraError(int &squereError){
		squereError = 0;
		for (size_t i = 0; i < tableTruthNumOfCol; ++i)
			squereError += vectorCalcFunction[i] ^ vectorTruthFunction[i];
	}

	// ������� �������� ������� ����� ������
	void calcNewValuesOfVARR(int delta, float etto = 0.3){
		for (size_t j = 0; j < sizePhiArr; j++){
			vArr[j] = vArr[j] + vDeltaArr[j];
			vDeltaArr[j] = static_cast<float>(etto * delta * phiArr[j]);
		}
		vArr[sizePhiArr] = vArr[sizePhiArr] + vDeltaArr[sizePhiArr];
		vDeltaArr[sizePhiArr] = static_cast<float>(etto * delta);
	}

	// ������� ������
	void printAnswer(int squereError, ofstream &out, size_t coutnEra){
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
			out << "), ";
			out << "E = " << squereError << "\n";
		}
	}

	// ������� �������� ������ � �������� ������������� �����
	void setNumOfBitAndNumWithMinNumOfBit(int num){
		size_t val = num;
		size_t bit = 0;
		while (val){
			if (val & 1)
				bit++;
			val >>= 1;
		}
		if (bit < minNumOfBit){
			minNumOfBit = bit;
			numWithMinNumOfBit = num;
		}
	}

	// �������, ������������ �������� �������� �������� �������
	void getTruthVectorOfValueForMySimpleFunc(){
		for (size_t i = 0; i < tableTruthNumOfCol; ++i)
			vectorTruthFunction[i] = ((xArr[0][i] | xArr[1][i]) & (!xArr[2][i]) & xArr[3][i]) == true ? 1 : 0;
	}

	// ����������� phiFunction
	float phiFunctionCalc(size_t col, bool *cArr){
		int degree = 0;
		for (size_t i = 0; i < tableTruthNumOfElement; ++i){
			degree += static_cast<int>(pow((static_cast<int>(xArr[i][col])) - static_cast<int>(cArr[i]), 2));
		}
		return static_cast<float>(exp(-degree));
	}

	// ����������� ������ ������� ������� cArr � ������� �� phiArr
	void calcSizePhiArr(){
		this->sizePhiArr = 0;
		this->oneOrZero = true;
		for (size_t i = 0; i < this->tableTruthNumOfCol; ++i)
			this->sizePhiArr += this->vectorTruthFunction[i];

		if (sizePhiArr == min(this->sizePhiArr, this->tableTruthNumOfCol - this->sizePhiArr))
			oneOrZero = true;
		else oneOrZero = false;

		this->sizePhiArr = min(this->sizePhiArr, this->tableTruthNumOfCol - this->sizePhiArr);
	}

	// �������, ������� �������� ������� ��� ��������� ������
	void setToZeroChangedVectors(){
		memset(this->phiArr, 0, sizePhiArr * sizeof(float));
		memset(this->vArr, 0, (sizePhiArr + 1) * sizeof(float));
		memset(this->vDeltaArr, 0, (sizePhiArr + 1) * sizeof(float));
		memset(this->vectorCalcFunction, 0, tableTruthNumOfCol * sizeof(size_t));
	}

	// ������� ����� �������� �������� ��� ������������ ������� 
	void training(int &currentNumWithMinNumOfBit, ofstream &out, bool print = false){
		size_t countEra = 1;
		int delta = 0; // ������� ������
		int squereError = -1; // ������ �����
		setToZeroChangedVectors();
		while (squereError && countEra++ < attempt){
			for (size_t i = 0; i < tableTruthNumOfCol; ++i){
				for (size_t j = 0; j < sizePhiArr; ++j){
					phiArr[j] = phiFunctionCalc(i, cArr[j]);
				}
				float net = netFunc(); // ������ net
				vectorCalcFunction[i] = outFunc(net); // ������� ��������
				delta = vectorTruthFunction[i] - vectorCalcFunction[i]; // ������ ������
				if ((1 << i) & currentNumWithMinNumOfBit)
					calcNewValuesOfVARR(delta); // ������������ ������ �����
			}
			calcValueOfEraError(squereError); // ���������� ������ � ���� ���
			if (out.is_open() && print)
				printAnswer(squereError, out, countEra);
		}
		if (countEra < attempt && !squereError && !print)
			setNumOfBitAndNumWithMinNumOfBit(currentNumWithMinNumOfBit);
		int tmp = 0;
		int count = 0;
		if (print){
			out << "������ ��������� ��������\n";
			while (currentNumWithMinNumOfBit){
				tmp = currentNumWithMinNumOfBit & 1;
				if (tmp)
					out << count << " ";
				count++;
				currentNumWithMinNumOfBit /= 2;
			}
		}
		
	}

public:
	VectorFunction(){}
	
	VectorFunction(size_t tableTruthNumOfElement){
		this->tableTruthNumOfElement = tableTruthNumOfElement,
			this->tableTruthNumOfCol = static_cast<size_t>(pow(2, tableTruthNumOfElement));
		this->xArr = new bool*[tableTruthNumOfElement];
		this->numWithMinNumOfBit = this->cFromNByK = static_cast<int>(pow(2, tableTruthNumOfCol)) - 1;
		this->minNumOfBit = tableTruthNumOfCol;

		for (size_t i = 0; i < tableTruthNumOfElement; ++i){
			this->xArr[i] = new bool[tableTruthNumOfCol];
		}

		int divider = 0;
		bool val = true;
		size_t j = 0;

		// ������� ������� � 4 �������� x1x2x3x4 � 16 ��������� 
		for (size_t i = 0; i < tableTruthNumOfElement; ++i){
			for (j = 0, divider = tableTruthNumOfCol / (static_cast<int>(pow(2, (i + 1)))), val = true; j < tableTruthNumOfCol; ++j){
				if (!((j) % divider))
					val = !val;
				this->xArr[i][j] = val;
			}
		}

		this->vectorTruthFunction = (size_t*)calloc(tableTruthNumOfCol, sizeof(size_t)); // ������ �������� �������� �������, ������� �� � ����� ������ ��������
		getTruthVectorOfValueForMySimpleFunc(); // ������ �������� �������� � ������ �������� �������
		calcSizePhiArr();
		this->phiArr = (float*)calloc(this->sizePhiArr, sizeof(float));
		this->cArr = new bool*[this->sizePhiArr];
		for (size_t i = 0; i < this->sizePhiArr; ++i)
			this->cArr[i] = new bool[this->tableTruthNumOfElement];
		size_t count = 0;
		for (size_t i = 0; i < this->tableTruthNumOfCol; ++i){
			if (oneOrZero && this->vectorTruthFunction[i] == 1){
				for (size_t j = 0; j < this->tableTruthNumOfElement; ++j){
					this->cArr[count][j] = this->xArr[j][i];
				}
				count++;
			}
			else if (!oneOrZero && this->vectorTruthFunction[i] == 0){
				for (size_t j = 0; j < this->tableTruthNumOfElement; ++j){
					this->cArr[count][j] = this->xArr[j][i];
				}
				count++;
			}
		}
		vArr = (float*)calloc(sizePhiArr + 1, sizeof(float)); // 00000 ���� ������
		vDeltaArr = (float*)calloc(sizePhiArr + 1, sizeof(float)); // ������ ������� ������ �������� 00000
		vectorCalcFunction = (size_t*)calloc(tableTruthNumOfCol, sizeof(size_t)); // ������ �������� ������� ��� ��������
	}

	~VectorFunction(){
		for (size_t i = 0; i < this->tableTruthNumOfElement; ++i){
			delete[]xArr[i];
		}
		for (size_t i = 0; i < this->sizePhiArr; ++i){
			delete[]cArr[i];
		}
		delete[]cArr;
		delete[]xArr;
		free(phiArr);
		free(vArr);
		free(vDeltaArr);
		free(vectorCalcFunction);
		free(vectorTruthFunction);
	}

	void start(ofstream &out){
		int currentNumWithMinNumOfBit = this->numWithMinNumOfBit - 1;
		for (int tr = 0; tr < this->cFromNByK; tr++){
			currentNumWithMinNumOfBit--;
			training(currentNumWithMinNumOfBit, out);
		}
		currentNumWithMinNumOfBit = this->numWithMinNumOfBit;
		training(currentNumWithMinNumOfBit, out, true);
	}
};

size_t VectorFunction::attempt = 10;

void main(int argc, char*argv[]){
	setlocale(LC_ALL, "rus");
	VectorFunction *vf = new VectorFunction(4);
	ofstream out;
	if (argc > 1)
		out.open(argv[1], ios::out | ios::trunc);
	else out.open("myout.txt");
	vf->start(out);
	delete vf;
}