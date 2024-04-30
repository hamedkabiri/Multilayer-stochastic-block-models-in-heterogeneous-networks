

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>
#include <ctime>
#include <vector>
#include<algorithm>
#include <sstream>
#include <typeinfo>
#include "HDCRMLSBMIntralayers.h"



using namespace std;
string trueCommsName;

int main(int argc, char *argv[])
{
    int N,K,i,j;
    N=100;
    K =1;
    
    double Values[N] ;
    double Aves[K] ;
    for (i=0 ; i<K ; i++)
    {
        Aves[i] = 0;
    }
    //ofstream outfile;
    //outfile.open("result.csv");
    
    for (i=0 ; i<K ; i++)
    {
        
        for (j=0 ; j<N ; j++)
        {
            Values[j] = 0;
        }
   
    std::string s = std::to_string(i);
    GetTheNetwork("Data" + s + ".csv","DataCommualpha=1.csv","DataType.csv");
    
    for (j=0 ; j<N ; j++)
    {
        Initialize();

        Values[j] = Apply_method_Restricted1();
  
        cout<<"ASLI= "<<Values[j]<<endl;
        Aves[i] += Values[j];
        PrintResults();
       
        ofstream outfile;
        std::string s = std::to_string(j);
        outfile.open("result" + s + ".csv");
        outfile<<Values[j];
        outfile.close();
    }
    //outfile<<endl;
    //Aves[i] = Aves[i]/N;
    //cout<<"Average="<<Aves[i]<<endl;
    }
    
    
    //for (i=0 ; i<K ; i++)
    //{
        //outfile<<Aves[i]<< ","; 
    //}
    //outfile<<endl;
    //outfile.close();
    return 0;
}
