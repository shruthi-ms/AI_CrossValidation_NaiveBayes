/*
AI ASSIGNMENT - 03 - (KNNC)
*/
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
//Declaration of global variables
int k = 3;
int kOptimumFound = 1;
double trainingSetArray[26][6670][193];
double testSetArray[3333][193];
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
                        //printf("incrementing ifold...\n");
                        //printf("%d--ifold before\n",ifold );
                        //printf("%d\n",k );
                        ifold = (ifold) % k;
                        //printf("%d---after\n",ifold );
                        if(ifold == 0){

                            ir++;
                            ir = ir%singleFoldRows;
                            //printf("%d %d \n",ifold,ir );
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

void showData(){

		int p,q,r;
		for(p=0;p<k;p++){
			for(q=0;q<singleFoldRows;q++){
				for(r=0;r<features;r++){
					printf("%lf ",trainingSetArray[p][q][r]);
				}
				printf("\n");
			}
		}
		for(p=0;p<testDataRows;p++){
			for(q=0;q<features;q++){
				printf("%lf ",testSetArray[p][q]);
			}
			printf("\n");
		}

}

double euclideanDistance(int index,int i,int j,int fold){

    int i1;
    double distance=0.0;
    for(i1=0;i1<features-1;i1++){

        distance+=(trainingSetArray[fold][index][i1]-trainingSetArray[i][j][i1])*(trainingSetArray[fold][index][i1]-trainingSetArray[i][j][i1]);
    }
    distance=sqrt(distance);
    return distance;
}
double euclideanDistancenew(int index,int i,int j){

    int i1;
    double distance=0.0;
    for(i1=0;i1<features-1;i1++){

        distance+=(testSetArray[index][i1]-trainingSetArray[i][j][i1])*(testSetArray[index][i1]-trainingSetArray[i][j][i1]);
    }
    distance=sqrt(distance);
    return distance;
}
void getSortedKNN(double euDistLabel[][2],int length){

	int i,j;
    double key,label;
    for(i=1;i<length;i++){
       
       key=euDistLabel[i][0];
       label=euDistLabel[i][1];
       j=i-1;
       while(j>=0 && euDistLabel[j][0]>key){
           
           euDistLabel[j+1][0]=euDistLabel[j][0];
           euDistLabel[j+1][1]=euDistLabel[j][1];
           j-=1;
       }
       euDistLabel[j+1][0]=key;
       euDistLabel[j+1][1]=label;
   	}
}


int getLabelnew(int index,int kOptimum,int fold){
	
	int i,j,k1=0;
	double euDistLabel[kOptimum+1][2];
	for(i=0;i<kOptimum+1;i++){

        euDistLabel[i][0]=-1.0;
        euDistLabel[i][1]=-1.0;
    }
    //Calculate KNNs for Test Instance
    double distance=0.0;
    for(i=0;i<5;i++){
        if(i!=fold){
            for(j=0;j<singleFoldRows;j++){

                if(trainingSetArray[i][j][features-1]!=-1.0){

                    //Calculating Euclidean distance.
                    distance=euclideanDistance(index,i,j,fold);
                    
                    //Finding kNNs on training set.
                    if(i==0 && j<=kOptimum){
                    
                        euDistLabel[k1][0]=distance;
                        euDistLabel[k1][1]=trainingSetArray[i][j][features-1];
                        k1++;
                        if(k1==(kOptimum+1)){

                            getSortedKNN(euDistLabel,kOptimum+1);
                        }
                    }
                    else{

                        if(distance < euDistLabel[kOptimum][0]){

                            euDistLabel[kOptimum][0]=distance;
                            euDistLabel[kOptimum][1]=trainingSetArray[i][j][features-1];
                            getSortedKNN(euDistLabel,kOptimum+1);
                        }
                    }
                }
            }
        }
    }
    //Predicting label on the basis of maximum count of NNs of a given class.
    int* labelCount=(int *)malloc(numOfClasses*sizeof(int));
    int max=0;
    int maxNNlabel=-1;
    for(i=0;i<numOfClasses;i++){

        labelCount[i]=0;
    }
    for(i=0;i<kOptimum;i++){

        labelCount[(int)euDistLabel[i][1]]+=1;
    }
    for(i=0;i<numOfClasses;i++){

        if(labelCount[i] > max){

            max=labelCount[i];
            maxNNlabel=i;
        }
    }

    return maxNNlabel;
}

double findmin(double a[]){
    double min=9999999999.0;
    int i;
    for(i=1;i<k;i++){
        if(a[i]<min){
            min = a[i];
        }
    }
    return min;
}

double kNNC(){
	int i,label,j;
    int correct,incorrect;
    double accuracy,error[26],mean_arr[26];
    int fold;
     for(j=1;j<6;j++){
        for(i=0;i<6;i++){
        error[i]=0;
        }
       for(fold=0;fold<k;fold++){
            correct=0;
            incorrect=0;
            for(i=0;i<testDataRows;i++){
        		if(testSetArray[i][features-1]!=-1.0){
             
                    label=getLabelnew(i,j,fold);
                    if(((int)testSetArray[i][features-1])==label){
                        correct += 1;
                    }  
                    else{
                        incorrect += 1;
                    }
                }
            }
            accuracy = ((correct*1.0))/(correct+incorrect);
            error[fold] = 1 - accuracy;
        }
        double mean=0;
        for(i=0;i<k;i++){
        //printf("%lf\n", error[i]);
            mean += error[i];
        }
        mean_arr[j] = mean/k;
        printf("%lf\n",mean);
    }
    for(i=1;i<6;i++){
        printf("%lf\n", mean_arr[i]);
    }
    double kmin = findmin(mean_arr);
    return ceil(kmin);
}
int getLabel(int index){
    
    int i,j,k1=0;
    double euDistLabel[kOptimumFound+1][2];
    for(i=0;i<kOptimumFound+1;i++){

        euDistLabel[i][0]=-1.0;
        euDistLabel[i][1]=-1.0;
    }
    //Calculate KNNs for Test Instance
    double distance=0.0;
    for(i=0;i<k;i++){

        for(j=0;j<singleFoldRows;j++){

            if(trainingSetArray[i][j][features-1]!=-1.0){

                //Calculating Euclidean distance.
                distance=euclideanDistancenew(index,i,j);
                
                //Finding kNNs on training set.
                if(i==0 && j<=kOptimumFound){
                
                    euDistLabel[k1][0]=distance;
                    euDistLabel[k1][1]=trainingSetArray[i][j][features-1];
                    k1++;
                    if(k1==(kOptimumFound+1)){

                        getSortedKNN(euDistLabel,kOptimumFound+1);
                    }
                }
                else{

                    if(distance < euDistLabel[kOptimumFound][0]){

                        euDistLabel[kOptimumFound][0]=distance;
                        euDistLabel[kOptimumFound][1]=trainingSetArray[i][j][features-1];
                        getSortedKNN(euDistLabel,kOptimumFound+1);
                    }
                }
            }
        }
    }

    //Predicting label on the basis of maximum count of NNs of a given class.
    int* labelCount=(int *)malloc(numOfClasses*sizeof(int));
    int max=0;
    int maxNNlabel=-1;
    for(i=0;i<numOfClasses;i++){

        labelCount[i]=0;
    }
    for(i=0;i<kOptimumFound;i++){

        labelCount[(int)euDistLabel[i][1]]+=1;
    }
    for(i=0;i<numOfClasses;i++){

        if(labelCount[i] > max){

            max=labelCount[i];
            maxNNlabel=i;
        }
    }

    return maxNNlabel;
}
double testUsingKOptimum(){
    int i,label;
    int correct=0,incorrect=0;
    double accuracy,error;
    for(i=0;i<testDataRows;i++){
        if(testSetArray[i][features-1]!=-1.0){
     
            label=getLabel(i);
            if(((int)testSetArray[i][features-1])==label){
                correct += 1;
            }  
            else{
                incorrect += 1;
            }
        }
    }
    accuracy = ((correct*1.0)*100.0)/(correct+incorrect);
    return accuracy;
}
int main(){
	//loading data from pp_tra.dat :)
	loadData();
	//printing data on terminal :P
	//showData();
	double kOptimum = kNNC();
    printf("kOptimum: %lf\n",kOptimum);
    kOptimumFound = kOptimum;
    double accuracy = testUsingKOptimum();
	printf("Accuracy: %lf\n",accuracy);
	return 0;
}