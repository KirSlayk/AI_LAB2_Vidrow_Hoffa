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
	// эра для текущего набора обучающих векторов 
	//не может быть больше, чем эта величина
	static size_t attempt;

	size_t tableTruthNumOfElement,
		tableTruthNumOfCol,
		sizePhiArr,
		minNumOfBit;
		
	// 2^16-1
	int cFromNByK,
		numWithMinNumOfBit;

	bool **xArr, // таблица истинности X-ов
		**cArr; // матрица векторов центров RBF-нейронов

	bool oneOrZero; // чего больше в векторе функции, нулей или единиц
	
	float *phiArr, // вектор значений гауссовой ФА
		*vArr, // вектор синаптических весов
		*vDeltaArr; // вектор дельта значений синамтических весов.  

	size_t *vectorTruthFunction, // вектор истинных значений заданной функции
		   *vectorCalcFunction;	// вектор значений функций для обучения
	
	// считатет net
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

	// функция подсчета ошибки эры
	void calcValueOfEraError(int &squereError){
		squereError = 0;
		for (size_t i = 0; i < tableTruthNumOfCol; ++i)
			squereError += vectorCalcFunction[i] ^ vectorTruthFunction[i];
	}

	// функция подсчета вектора весов связей
	void calcNewValuesOfVARR(int delta, float etto = 0.3){
		for (size_t j = 0; j < sizePhiArr; j++){
			vArr[j] = vArr[j] + vDeltaArr[j];
			vDeltaArr[j] = static_cast<float>(etto * delta * phiArr[j]);
		}
		vArr[sizePhiArr] = vArr[sizePhiArr] + vDeltaArr[sizePhiArr];
		vDeltaArr[sizePhiArr] = static_cast<float>(etto * delta);
	}

	// функция вывода
	void printAnswer(int squereError, ofstream &out, size_t coutnEra){
		if (out.is_open()){
			out << "Номер эпохи " << coutnEra << "\t";
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

	// функция подсчета единиц в двоичном представлении числа
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

	// функция, возвращающая истинное значение выданной функции
	void getTruthVectorOfValueForMySimpleFunc(){
		for (size_t i = 0; i < tableTruthNumOfCol; ++i)
			vectorTruthFunction[i] = ((xArr[0][i] | xArr[1][i]) & (!xArr[2][i]) & xArr[3][i]) == true ? 1 : 0;
	}

	// высчитывает phiFunction
	float phiFunctionCalc(size_t col, bool *cArr){
		int degree = 0;
		for (size_t i = 0; i < tableTruthNumOfElement; ++i){
			degree += static_cast<int>(pow((static_cast<int>(xArr[i][col])) - static_cast<int>(cArr[i]), 2));
		}
		return static_cast<float>(exp(-degree));
	}

	// высчитывает размер вектора центров cArr и вектора ФА phiArr
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

	// функция, которая обнуляет векторы для повторной работы
	void setToZeroChangedVectors(){
		memset(this->phiArr, 0, sizePhiArr * sizeof(float));
		memset(this->vArr, 0, (sizePhiArr + 1) * sizeof(float));
		memset(this->vDeltaArr, 0, (sizePhiArr + 1) * sizeof(float));
		memset(this->vectorCalcFunction, 0, tableTruthNumOfCol * sizeof(size_t));
	}

	// функция всего процесса обучения для определенной выборки 
	void training(int &currentNumWithMinNumOfBit, ofstream &out, bool print = false){
		size_t countEra = 1;
		int delta = 0; // текущая дельта
		int squereError = -1; // ошибка эпохи
		setToZeroChangedVectors();
		while (squereError && countEra++ < attempt){
			for (size_t i = 0; i < tableTruthNumOfCol; ++i){
				for (size_t j = 0; j < sizePhiArr; ++j){
					phiArr[j] = phiFunctionCalc(i, cArr[j]);
				}
				float net = netFunc(); // считаю net
				vectorCalcFunction[i] = outFunc(net); // процесс обучения
				delta = vectorTruthFunction[i] - vectorCalcFunction[i]; // считаю дельту
				if ((1 << i) & currentNumWithMinNumOfBit)
					calcNewValuesOfVARR(delta); // пересчитываю вектор весов
			}
			calcValueOfEraError(squereError); // высчитываю ошибку в этой эре
			if (out.is_open() && print)
				printAnswer(squereError, out, countEra);
		}
		if (countEra < attempt && !squereError && !print)
			setNumOfBitAndNumWithMinNumOfBit(currentNumWithMinNumOfBit);
		int tmp = 0;
		int count = 0;
		if (print){
			out << "Номера обучающих векторов\n";
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

		// забиваю матрицу с 4 строками x1x2x3x4 и 16 столбцами 
		for (size_t i = 0; i < tableTruthNumOfElement; ++i){
			for (j = 0, divider = tableTruthNumOfCol / (static_cast<int>(pow(2, (i + 1)))), val = true; j < tableTruthNumOfCol; ++j){
				if (!((j) % divider))
					val = !val;
				this->xArr[i][j] = val;
			}
		}

		this->vectorTruthFunction = (size_t*)calloc(tableTruthNumOfCol, sizeof(size_t)); // вектор истинных значений функции, которые мы в итоге должны получить
		getTruthVectorOfValueForMySimpleFunc(); // заношу истинные значения в вектор значений функции
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
		vArr = (float*)calloc(sizePhiArr + 1, sizeof(float)); // 00000 веса связей
		vDeltaArr = (float*)calloc(sizePhiArr + 1, sizeof(float)); // вектор текущих дельта значений 00000
		vectorCalcFunction = (size_t*)calloc(tableTruthNumOfCol, sizeof(size_t)); // вектор значений функций для обучения
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