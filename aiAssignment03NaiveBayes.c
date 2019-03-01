/*
AI ASSIGNMENT - 03 - (Naive Bayes)
*/
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
//Declaration of global variables
int k = 3;
int kOptimum = 1;
double trainingSetArray[26][6670][193];
double testSetArray[3333][193];
double NaiveBayesValues[10][193][5];
double classes[3333];
double pOfClass[11];
double pOfWiByX[11];
int trainDataRows = 6670;
int testDataRows = 3333;
int features = 193;
int singleFoldRows;
int numOfClasses = 10;
//loading data from file
void loadData(){
	printf("loading train data.....\n");
	char trainData[] ="C:/Users/M.S.Shruthi/Downloads/pp_tra.dat";
	char testData[] = "C:/Users/M.S.Shruthi/Downloads/pp_tes.dat";
	FILE* fp;
    //Loading training set data
	fp = fopen(trainData,"r");
	singleFoldRows = ceil((1.0*trainDataRows)/k);
	int i, c, ifold=0, ir=0, ic=0, iv=0;
    char value[10];
    if(fp){
    	//printf("in if condition...\n");
        while((c=getc(fp)) != EOF){
            if(c == ' ' || c == '\n'){
                    value[iv] ='\0';
                    trainingSetArray[ifold][ir][ic]=atoi(value);
                    ic++;
                    ic = ic%features;
                    if(ic == 0){
                        ifold++;
                        ifold = (ifold) % k;
                        if(ifold == 0){
                            ir++;
                            ir = ir%singleFoldRows;
                        }
                    }
                    iv = 0;
                }
            else{
                value[iv] = c;
                iv++;
                iv = iv%10;
            }
        }
        while(ifold != 0){
            for(i = 0;i < features;i++){
                trainingSetArray[ifold][ir][i]=-1.0;
            }
            ifold++;
            ifold = ifold%k;
        }
    }    
    //Loading test data
    printf("loading test data.....\n");
    fp = fopen(testData,"r");
    c=0;ir=0;ic=0;iv=0;
    if(fp){
        while((c=getc(fp))!=EOF){
            if(c == ' ' || c == '\n'){            
                value[iv] = '\0';
                testSetArray[ir][ic] = atoi(value);
                ic++;
                ic = ic%features;
                if(ic == 0){
                    ir++;
                    ir = ir%testDataRows;
                }
                iv = 0;
            }
            else{
                value[iv] = c;
                iv++;
                iv = iv%10;
            }
        }
    }
    printf("loaded.\n");
}

double getProbability(int index){
    int count = 0;
	for (int i = 0; i < k; i++){
        for (int j = 0; j < singleFoldRows; j++){
            if(trainingSetArray[i][j][features-1]==index) count ++;
        }
    }
    return (float)count/trainDataRows;
}

void calculatePriorityOfClasses(){
    for (int i = 0; i < numOfClasses; i++) pOfClass[i] = getProbability(i);    
}

void calculateLikelihood(int class){
    for (int i = 0; i < k; i++){
        for (int j = 0; j < singleFoldRows; j++){
            if(trainingSetArray[i][j][features-1]==class){
                for (int p = 0; p < features-1; p++){
                    NaiveBayesValues[class][p][(int)trainingSetArray[i][j][p]] ++;                
                }
            }
        }
    }
}

void checktest(int index){
    int p,q,r;
    double class[11];
    double x;
   for (p = 0; p < numOfClasses; p++){
       for (q = 0; q < features-1; q++){
            x = x +  log(NaiveBayesValues[p][q][(int)testSetArray[index][q]]);
       }
       class[p] = x;
       x = 0;
   }
   double m,max = -9999999.0;
   for(p=0;p<10;p++){
        if(class[p] > max){
            max = class[p];
            m = (double)p;
        }
    }
    classes[index] = m;
} 

double NaiveBayes(){
    calculatePriorityOfClasses();
    int i,p,q,r;
    for(i=0;i<numOfClasses;i++){
        calculateLikelihood(i);
    }
    //calc prob
    for(p = 0 ; p < numOfClasses ; p++){
        for(q = 0 ; q <features-1 ; q++){
            for(r = 0 ; r < 5; r++){
                NaiveBayesValues[p][q][r] = (NaiveBayesValues[p][q][r])/(pOfClass[p]*trainDataRows);
            }
        }
    }   
    for(p=0;p<testDataRows;p++){
       checktest(p);
    }
    int correct=0,incorrect=0;
    for(i=0;i<testDataRows;i++){
        if(classes[i] == testSetArray[i][192]){
            correct += 1;
        }
        else{
            incorrect += 1;
        }
    }
    printf("%d\n",correct );
	return ((float)correct/testDataRows)*100;
}
int main(){
	//loading data from pp_tra.dat and pp_tes.dat :)
	loadData();
	//printing data on terminal :P
	double accuracy = NaiveBayes();
	printf("accuracy: %lf\n",accuracy );
	return 0;
}
