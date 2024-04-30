
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
#include <typeinfo>
#include <vector>
#include<algorithm>
#include <sstream>
using namespace std;

// /* const values to control maximum sizes */
const int MaxComms= 5 ;
const long int Nodes = 300;
const int Layers = 3;
const bool DegreeCorrect = 1;
const int MAXEDGES = 10000000;  // this is the maximum number of edges

/* empty global declarations */

 // 0 means don't correct for the degrees, 1 means do correct for the degrees.
 int  Comms[Nodes];
 int Types[Nodes];
 int *AdjMatrix[Nodes];
 int degg[Nodes];
/* Number of random initializations (default) */
unsigned int KLPerNetwork = 1;

/********** GLOBAL VARIABLES **********/
bool TrueCommsAvailable = 1; // (default) Changes to 1 if user passes in true comms.
bool InitializationOption = 0; // (default) May be changed to 1 by user.

// OVERALL GENERAL VARIABLES NEEDED FOR ALL OF THE CODE

long int Degree[Nodes][Layers];  // Degree of nodes in the network
 // The first is the maximum number of edges to read in
double LL[Layers][Layers];
double NL[Layers];
// FOR KL
unsigned int CurrentState[Nodes];
int BestState[Nodes];
int ChangeSet[Nodes];
int UpdateIndex[Nodes];
int TrueState[Nodes]; // This records the true communities if they exist read in from the file


double MaxScore = 0;

int BestCommVertices[Layers][MaxComms];
int BestCommStubs[Layers][Layers][MaxComms];
int BestEdgeMatrix[Layers][Layers][MaxComms][MaxComms];

int CurrentCommVertices[Layers][MaxComms];
int CurrentCommStubs[Layers][Layers][MaxComms];
int CurrentEdgeMatrix[Layers][Layers][MaxComms][MaxComms];

double CurrentCommVerticesNew[MaxComms];
  double CurrentCommStubsNew[MaxComms];
  double CurrentEdgeMatrixNew[MaxComms][MaxComms];





int TempNeighborSet[Layers][2][MaxComms];  // the first entry lists the comm and the second entry lists the number of edges to that comm
int NeighborSet[Layers][2][MaxComms]; // this is what we record and use
int SelfEdgeCounter = 0; // this records self-edges to make sure that they are counted correctly
int ActualDiffComms[Layers] ; // this records the number of different comms in neighborhood
int NeighborLayers[Layers] ;
// For reporting best state
int SavedState[Nodes];
int SavedCommVertices[Layers][MaxComms];
int SavedCommStubs[Layers][Layers][MaxComms];
int SavedEdgeMatrix[Layers][Layers][MaxComms][MaxComms];
double NMIValue = 0;
double VIValue = 0;
double HighestScore = 0;

/* function declarations */
//void freegraph(); // gets rid of the graph pointers at end
void GetTheNetwork(string fileName1,string fileName2,string fileName3);
void RunKL(); // runs Kernighan-Lin once.
void Initialize(); // initializes the data structures for KL
double ComputeInitialScore(); // computes the initial score after initialization
void ComputeNeighborSet(int vertex, int option);  // computes the neighbor set for a vertex, using either best or currentstates
double ComputeProposal(int vertex, int from, int destination,int fromLayer); // computes the value of the particular change
void UpdateMatrices(int vertex, int option, int from, int destination,int fromLayer); // this updates either the best
//or current matrices based on moving the vertex from from to destination
double LogFunction(double x); // this returns x*log(x) and zero if x=0
void PrintResults(); // prints out the resulting graph for now
double ComputeVI();
double ComputeNMI();
double Entropy(int entoption);
double JointEntropy();
double Apply_method_Restricted3();
int maxx(std::vector<int> x);


double Apply_method_Restricted3()
{
  HighestScore = -numeric_limits<double>::max( );
  VIValue = 0;
  NMIValue = 0;
 
  unsigned int i,j,k,m,l;
   for(j=0; j < KLPerNetwork; j++)
   {
     RunKL();
     //KL,dt,L:
     
    if(MaxScore >= HighestScore)
    {
      HighestScore = MaxScore;
      if(TrueCommsAvailable == 1)
      {
	VIValue = ComputeVI();
	NMIValue = ComputeNMI();
	cout << "VI Value: " << VIValue << " NMI Value: " << NMIValue << endl;
      }
      for(m=0; m < Layers; m++)
          for(l=0; l < Layers; l++)
              for(i=0; i < MaxComms; i++)
              {
	          SavedCommVertices[m][i] = BestCommVertices[m][i];
	          SavedCommStubs[m][l][i] = BestCommStubs[m][l][i];
	          for(k=0; k < MaxComms; k++)
	              SavedEdgeMatrix[m][l][i][k] = BestEdgeMatrix[m][l][i][k];
               }
      for(i=0; i < Nodes; i++)
      {
	SavedState[i] = BestState[i];
      }
    }
  }
  
  // because PrintResults are written for best values we copy them
  // back over from the saved values which are the best ones.
  for(m=0; m < Layers; m++)
          for(l=0; l < Layers; l++)
             for(i=0; i < MaxComms; i++)
             {
                   BestCommVertices[m][i] = SavedCommVertices[m][i];
                   BestCommStubs[m][l][i] = SavedCommStubs[m][l][i];
                   for(k=0; k < MaxComms; k++)
                       BestEdgeMatrix[m][l][i][k] = SavedEdgeMatrix[m][l][i][k];
              }
  for(i=0; i < Nodes; i++)
  {
    BestState[i] = SavedState[i];
  }
  cout << "Final Score: " << ComputeInitialScore() << endl;
  
  for(m=0; m < Layers; m++)
 {
      for(l=0; l < Layers; l++)
      {
         for(i=0; i < MaxComms; i++)
         {
             for(j=0; j < MaxComms; j++)
                  cout<< BestEdgeMatrix[m][l][i][j]<<' ';
             cout<<endl; 
         }
         cout<<endl; 
      }
      cout<<endl; 
 }
  
  //for(i=0; i < Nodes; i++)
     //delete [] AdjMatrix[i];
  //freegraph();
  
  return NMIValue;
}



// This function deletes the graph from memory.
//void freegraph()
//{
 // long int i;//-Wunused,j;
  
  //for(i=0; i < Nodes; i++)
   // delete [] AdjList[i];
  
  //return;
//}

void RunKL()
{
  unsigned int i,j,k,m,l;
  int MaxIndex = 1;
  double CurrentScore;  // records the current log-likelihood
  int MaxVertex;  // this records the index of the largest vertex ratio found so far
  double MaxRatio;  // records the value of the ratio, actually it's the log of the ratio
  int MaxPriority; // records the community that the vertex wants to go to
  long int tempvertex = 0;
  
  double prevMaxScore = -numeric_limits<double>::max( );
  long double tolerance = 0.00000000001; // this prevents loops due to numerical errors.
  
  double ProposalRatio;
  double value;
  int Priority;
  
  
  
  // This returns the log of the initial score
  MaxScore = ComputeInitialScore();
  int iter = 0;
  int maxIter = 100;
  while(MaxScore >= prevMaxScore + tolerance && iter < maxIter)
  {
    iter++;
    cout << "MAX SCORE IS: " << MaxScore << endl;
    // we start with everything equal to the best values
    CurrentScore = MaxScore;
    prevMaxScore = MaxScore;
    MaxIndex = -1;
    
    // ChangeSet records which vertices are able to move, in that they haven't
    // already moved during this KL step.  Update index will tell when the vertex
    // was chosen to move.
    for(i=0; i < Nodes; i++)
    {
      CurrentState[i] = BestState[i];
      ChangeSet[i] = i;
      UpdateIndex[i] = -1;
    }
    
    for(m=0; m < Layers; m++)
          for(l=0; l < Layers; l++)
             for(i=0; i < MaxComms; i++)
             {
                  CurrentCommVertices[m][i] = BestCommVertices[m][i];
                  CurrentCommStubs[m][l][i] = BestCommStubs[m][l][i];
                  for(j=0; j < MaxComms; j++)
	               CurrentEdgeMatrix[m][l][i][j] = BestEdgeMatrix[m][l][i][j];
             }
    
    // This loop moves each vertex once
    // Note that we DONT reinitialize changeset as this is unnecessary
    // This would make it a factor of 2 slower.
    for(i=0; i < Nodes; i++)
    {
      MaxVertex = 0;
      MaxRatio = -numeric_limits<double>::max( );
      MaxPriority = 0;
      // This loop selects which vertex to move
      for(j=0; j < Nodes-i; j++)
      {
	// get proposal and proposal ratio for ChangeSet[j]
	Priority = 0;
	ProposalRatio = -numeric_limits<double>::max( );
	// we first compute the neighbor set of the vertex, this is fixed
	// and the same for every change,
	// computing this first makes this more efficient
	// zero indicates run with current communities
	ComputeNeighborSet(ChangeSet[j], 0);
	
	// DanLarremore Modification:
	// We actually don't want to try all the comms, because some of them are forbidden.
	// We only get to choose from the entries of Comms[Types[j]].
	
	for (unsigned int k=0; k < MaxComms ; ++k)
	{
	  
	  // we compute the value of vertex ChangeSet[j] going to k
	  // we DONT allow a vertex to remain where it was
	  // This is essential to enforce so that it will go downhill and not be greedy
	  if(k != CurrentState[ChangeSet[j]])
	  {
	    value = ComputeProposal(ChangeSet[j], CurrentState[ChangeSet[j]], k,Types[ChangeSet[j]]);
	    if(value > ProposalRatio)
	    {
	      Priority = k;
	      ProposalRatio = value;
	    }
	  }
	}
	
	// check whether its higher than what you already have as the max KL move
	if(ProposalRatio > MaxRatio)
	{
	  MaxVertex = j;  // Note this is not the vertex j, but the vertex given by ChangeSet[j]
	  MaxRatio = ProposalRatio;
	  MaxPriority = Priority;
	}
      }
      // now we move it, first recording the current neighbors so that
      // we can update the matrices properly
      ComputeNeighborSet(ChangeSet[MaxVertex], 0);
      // This updates the matrices to represent the vertices new state
      UpdateMatrices(ChangeSet[MaxVertex], 0, CurrentState[ChangeSet[MaxVertex]], MaxPriority,Types[ChangeSet[MaxVertex]]);
      CurrentState[ChangeSet[MaxVertex]] = MaxPriority;
      // we are using logs so we add the maxratio to the current score for the new score
      CurrentScore = CurrentScore + MaxRatio;
      UpdateIndex[ChangeSet[MaxVertex]] = i;
      // we switch it with the last element of changeset, removing it from further consideration
      // until we have moved the other vertices
      tempvertex = ChangeSet[MaxVertex];
      ChangeSet[MaxVertex] = ChangeSet[Nodes-i-1];
      ChangeSet[Nodes-i-1] = tempvertex;
      
      // now if the new state is better than the previous best state we record this
      // MaxIndex in combination with UpdateIndex
      // telling us where we are in the movement of vertices
      if(CurrentScore > MaxScore)
      {
	MaxScore = CurrentScore;
	MaxIndex = i;
      }
    }
    
    
    // now we update BestState if a change resulted in a higher maximum
    // by implementing all the changes found above
    
    // There is a potential for speeding this up here.
    if(MaxIndex != -1)
    {
      for(i=0; i < Nodes; i++)
      {
	// we only make the changes to beststate that happened before or equal to maxindex
	// no other vertex is updated
	// fortunately the update order is irrelevant to the final result so
	// we can just do it this way
	// Because we force all moves to be different, these updates are all a switch of community
	if(UpdateIndex[i] <= MaxIndex)
	{
	  // the option 1 does update corresponding to the best states / matrices
	  ComputeNeighborSet(i, 1);
	  UpdateMatrices(i, 1, BestState[i], CurrentState[i],Types[i]); // 1 does best matrix update
	  BestState[i] = CurrentState[i];
	}
      }
    }
    
  }
  if (iter==maxIter)
  {
    cout << "...maxIterations on this KL run." << endl;
  }
  
  return;
}

// This starts off from a random initial condition
void Initialize()
{ 
    cout<<"khar";
  unsigned int i, j,m,l;
  int neighbor;
  int sum;
  MaxScore = 0;
  NMIValue = 0;
  VIValue = 0;
  HighestScore = 0;
  
  for(i=0; i < Nodes ; i++)
      TrueState[i] = 0;
  
  for(i=0; i < Nodes ; i++)
  {
      TrueState[i] = Comms[i];
      CurrentState[i] = 0;
      BestState[i] = 0;
      ChangeSet[i] = 0;
      UpdateIndex[i] = 0;
      SavedState [i] = 0;
  }
  
  for(l=0; l < Layers ; l++)
  {
      
      NL[l] = 0;
      for(m=0; m < Layers ; m++)
          LL[m][l] = 0;
      for(i=0; i < Nodes ; i++)
          Degree[i][l] = 0 ;
  }
  
  for(i=0; i < Nodes ; i++)
  {     

      NL[Types[i]] += 1;
      for(j=0; j < degg[i] ; j++)
      {
           m = Types[i];
           l = Types[AdjMatrix[i][j]];
           LL[m][l] += double(1);
           Degree[i][l] += 1;
      }
  }
  
  
  
  for(m=0; m < Layers; m++)
      for(l=0; l < Layers; l++)
         for(i=0; i < MaxComms; i++)
           {
             BestCommVertices[m][i] = 0;
             CurrentCommVertices [m][i] = 0;
             SavedCommVertices [m][i] = 0;
             BestCommStubs[m][l][i] = 0;
             CurrentCommStubs [m][l][i] = 0;
             SavedCommStubs [m][l][i] = 0;
             for(j=0; j < MaxComms; j++)
             {
                 BestEdgeMatrix[m][l][i][j] = 0;
                 CurrentEdgeMatrix[m][l][i][j] = 0;
                 SavedEdgeMatrix[m][l][i][j] = 0;
             }
            }
  
     

      
  srand(time(0));
  for(i=0; i < Nodes; i++)
  {
    // BestState[i] = int(numgen.nextDouble(MaxComms));   // REPLACERNG, should return 0 to MaxComms-1 in integer
    // DanLarremore Modification:
    // The initialized communities must be constrained to respect types.
    // cout << i << "," << Types[i] << endl;
    BestState[i] =  (rand() % MaxComms );
    m = Types[i];
    if(InitializationOption == 1)
           BestState[i] = TrueState[i];
    BestCommVertices[m][BestState[i]]++;
    
    for(l=0; l < Layers; l++)
        BestCommStubs[m][l][BestState[i]] += Degree[i][l];
  }
  cout<<"BestState=";
  for(i=0; i < Nodes; i++)
      cout<<BestState[i]<<' ';
  cout<<endl;
  // We are going to double count all edges and then divide two
  for(i=0; i < Nodes; i++)
  {
    for(j=0; j < degg[i]; j++)
    {
      
      BestEdgeMatrix[Types[i]][Types[AdjMatrix[i][j]]][BestState[i]][BestState[AdjMatrix[i][j]]] += 1;
      
    }
  }
  
  
      return;
}

double ComputeInitialScore()
{
  // For the running of the KL algorithm itself this does not matter as all we use
  // are whether the score increases
  // We will want this when we compare different initializations
  
  // this actually returns 1/2 the unnormalized log-likelihood listed in the paper
  
  unsigned int i;
  unsigned int j, m,l, p ,r;
  double sum = 0;
  double kab = 1;
  double BestCommVerticesNew[MaxComms];
double BestCommStubsNew[MaxComms];
double BestEdgeMatrixNew[MaxComms][MaxComms];
    for (i=0 ; i<MaxComms ; i++)
  {
         BestCommVerticesNew[i] = 0;
         BestCommStubsNew[i]=0;
         for (j=0 ; j<MaxComms ; j++)
             BestEdgeMatrixNew[i][j]=0;
  }
  for (m=0 ; m<Layers ; m++)
      for (i=0 ; i<MaxComms ; i++)
      {
       
         for (l=m ; l<Layers ; l++)
            if(LL[m][l]!=0 && LL[m][l]!=0 )
             {
                if(m!=l)
                {
                    kab=1;
                    for (p=0 ; p<Layers ; p++)
                        for (r=p ; r<Layers ; r++)
                            if(LL[p][r]!=0 && LL[r][p]!=0)
                              if(p!=m || r!=l)
                                if(p!=r)
                                    kab = kab*(LL[p][r]+LL[r][p]);
                                else
                                    kab = kab*(LL[p][r]);
                    
                    for (j=0 ; j<MaxComms ; j++)
                                BestEdgeMatrixNew[i][j] += (double(BestEdgeMatrix[m][l][i][j]+BestEdgeMatrix[l][m][i][j])*kab);
                }
                else
                {
                    kab=1;
                    for (p=0 ; p<Layers ; p++)
                        for (r=p ; r<Layers ; r++)
                            if(LL[p][r]!=0 && LL[r][p]!=0)
                              if(p!=m || r!=l)
                                if(p!=r)
                                    kab = kab*(LL[p][r]+LL[r][p]);
                                else
                                    kab = kab*(LL[p][r]);
                 
                    for (j=0 ; j<MaxComms ; j++)
                          BestEdgeMatrixNew[i][j] += (double(BestEdgeMatrix[m][l][i][j])*kab);
                }
             }
      }
        
    kab=1;
    for (p=0 ; p<Layers ; p++)
        for (r=p ; r<Layers ; r++)
            if(LL[p][r]!=0 && LL[r][p]!=0)
             
                    if(p!=r)
                        kab = kab*(LL[p][r]+LL[r][p]);
                    else
                        kab = kab*(LL[p][r]);
  
  
     
     for (i=0 ; i<MaxComms ; i++)
     {
         
      
         for (j=0 ; j<MaxComms ; j++)
             BestEdgeMatrixNew[i][j]= (BestEdgeMatrixNew[i][j]/kab);
     }
    
    
    ///cout<<"bbbbb"<<endl;
    //double kkk=0;
            //for (i=0 ; i<MaxComms ; i++)
                //{
                  //for (j=0 ; j<MaxComms ; j++)
                  //{
                            // kkk+= BestEdgeMatrixNew[i][j];
                            // cout<<BestEdgeMatrixNew[i][j]<<' ';
                  //}
                 // cout<<endl;
              //    }
    //cout<<kkk<<endl;
    
    
  
  
        for(m=0; m < Layers; m++)
        {
            for(l=0; l < Layers; l++)
             
               if (LL[m][l]!=0 && LL[l][m]!=0)
                  for(i=0; i < MaxComms; i++)
                  {

                         if (DegreeCorrect == 1)
                             if(m!=l)
                                sum = sum - 2*LogFunction(double(BestCommStubs[m][l][i]+BestCommStubs[m][l][i])/(LL[m][l]+LL[l][m]));
                             else
                                 sum = sum - 2*LogFunction(double(BestCommStubs[m][l][i])/(LL[m][l]));
                  }
        }
        for(i=0; i < MaxComms; i++)
        {
            
            for(j=0; j < MaxComms; j++)
                 sum = sum + LogFunction(double(BestEdgeMatrixNew[i][j]));
        }
  
        return sum;
}

// We compute this using the current comm matrices
// We avoid the potential pitfalls of huge intermediate numbers by adding logs together.  So we treat 0 log 0 as 0.
// We return 0 for degree zero vertices (which really shouldn't be sent into the program
// in the first place.)
// We also return 0 for from = destination cause there is no change then.
// Here we use base e.  It returns the log of the actual value.
// Again this is half of the change in the unnormalized log-likelihood listed in the paper
double ComputeProposal(int vertex, int azz, int destination,int fromLayer)
{
  unsigned int i,j,r;//-Wunused, j, k;
  unsigned int l,p,m;
  double ratiovalue = 0;
  double fromcount = 0;
  double destcount = 0;
  double kab=1;
  double help1;
  double help2;
  double help3[MaxComms];
  
  if(azz == destination)
    return 0;
  
  // if the degree of the vertex is zero we know nothing about it
    // in this case we don't ever change its community
    // at the end we put all degree zeroes into their own group
            
  
     for (i=0 ; i<MaxComms ; i++)
  {
         CurrentCommVerticesNew[i] = 0;
         CurrentCommStubsNew[i]=0;
         for (j=0 ; j<MaxComms ; j++)
             CurrentEdgeMatrixNew[i][j]=0;
  }
  for(m=0 ; m < Layers; m++)
        for(l=m ; l < Layers; l++)
            if(LL[m][l]!=0 && LL[l][m]!=0)
            for (i=0 ; i<MaxComms ; i++)
                if(m!=l)
                {
                CurrentCommStubsNew[i]+=(double(CurrentCommStubs[m][l][i]+CurrentCommStubs[l][m][i])/(LL[m][l]+LL[l][m]));
                for (p=0 ; p<MaxComms ; p++)
                    CurrentEdgeMatrixNew[i][p]+=(double(CurrentEdgeMatrix[m][l][i][p]+CurrentEdgeMatrix[l][m][i][p])/(LL[m][l]+LL[l][m]));
                }
                else
                {
                    CurrentCommStubsNew[i]+=(double(CurrentCommStubs[m][l][i])/(LL[m][l]));
                    for (p=0 ; p<MaxComms ; p++)
                    CurrentEdgeMatrixNew[i][p]+=(double(CurrentEdgeMatrix[m][l][i][p])/(LL[m][l]));
                }
  //cout<<"matrix"<<endl;
  //for (i=0 ; i<MaxComms ; i++)
  //{
    //  for (p=0 ; p<MaxComms ; p++)
    //      cout<<CurrentEdgeMatrixNew[i][p]<<' ';
   //    cout<<endl;
  //}
 // for (m=0 ; m<Layers ; m++)
  //    for (i=0 ; i<MaxComms ; i++)
   //   {
       
   //      for (l=m ; l<Layers ; l++)
    //        if(LL[m][l]!=0 && LL[m][l]!=0 )
     //        {
    //            if(m!=l)
    //            {
    //                kab=1;
    //                for (p=0 ; p<Layers ; p++)
    //                    for (r=p ; r<Layers ; r++)
    //                        if(LL[p][r]!=0 && LL[r][p]!=0)
    //                          if(p!=m || r!=l)
    //                            if(p!=r)
    //                                kab = kab*(LL[p][r]+LL[r][p]);
    //                            else
    //                                kab = kab*(LL[p][r]);
                    
   //                 for (j=0 ; j<MaxComms ; j++)
   //                             CurrentEdgeMatrixNew[i][j] += (double(CurrentEdgeMatrix[m][l][i][j]+CurrentEdgeMatrix[l][m][i][j])*kab);
   //             }
   //             else
   //             {
   //                 kab=1;
   //                 for (p=0 ; p<Layers ; p++)
   //                     for (r=p ; r<Layers ; r++)
   //                         if(LL[p][r]!=0 && LL[r][p]!=0)
    //                          if(p!=m || r!=l)
    //                            if(p!=r)
   //                                 kab = kab*(LL[p][r]+LL[r][p]);
   //                             else
   //                                 kab = kab*(LL[p][r]);
                 
   //                 for (j=0 ; j<MaxComms ; j++)
   //                       CurrentEdgeMatrixNew[i][j] += (double(CurrentEdgeMatrix[m][l][i][j])*kab);
  //              }
  //           }
  //    }
        
  //  kab=1;
  //  for (p=0 ; p<Layers ; p++)
     //   for (r=p ; r<Layers ; r++)
         //   if(LL[p][r]!=0 && LL[r][p]!=0)
             
          //          if(p!=r)
          //              kab = kab*(LL[p][r]+LL[r][p]);
         //           else
           //             kab = kab*(LL[p][r]);
  
  
     
     //for (i=0 ; i<MaxComms ; i++)
     //{
         
      
        // for (j=0 ; j<MaxComms ; j++)
         //    CurrentEdgeMatrixNew[i][j]= (CurrentEdgeMatrixNew[i][j]/kab);
     //}
    
  
       long deg = 0;
        for(l=0 ; l < Layers; l++)
            deg += Degree[vertex][l];

        if (DegreeCorrect == 1)

           if (deg == 0)
                return 0;
         for (p=0 ; p<MaxComms ; p++)
             help3[p]=0;
        
        fromcount = 0;
        destcount = 0;
        for(l=0 ; l < Layers; l++)
             if (LL[fromLayer][l] != 0 && l!=fromLayer)
                 if (NeighborLayers[ l] == 1)
                 {
                    
                    for (i=0 ; i<ActualDiffComms[l] ; i++)
                    {

                           if ((NeighborSet[ l ] [0 ][ i] != azz) && (NeighborSet[ l][0 ][i] != destination))
                           {
                               
                                help3[NeighborSet[ l ] [0 ][ i]] += (double(NeighborSet[l ][1 ][i])/(LL[fromLayer][l]+LL[l][fromLayer]));
                      
                           }

                           if (NeighborSet[l ][0 ][i] == azz)
                               
                                  fromcount += (double(NeighborSet[l ][1 ][i])/(LL[fromLayer][l]+LL[l][fromLayer]));
                               
                           if (NeighborSet[l ][0 ][i] == destination)
                                
                                  destcount += (double(NeighborSet[l ][1 ][i])/(LL[fromLayer][l]+LL[l][fromLayer]));
                              
                                    
                    }
                    
                               //cout<<"kabiri5"<<endl;
                               help1 = double(CurrentCommStubs[fromLayer ][l ][ azz]+CurrentCommStubs[l ][fromLayer ][ azz]);
                               help2 = double(Degree[vertex ][l]);
                               ratiovalue = ratiovalue - 2*LogFunction((help1 - help2)/(LL[fromLayer][l]+LL[l][fromLayer]) ) + 2*LogFunction(help1/(LL[fromLayer][l]+LL[l][fromLayer]) );
                        
                       
      

                   
                        
                           //cout<<"kabiri9"<<endl;
                            help1 = double(CurrentCommStubs[fromLayer ][l ][destination]+CurrentCommStubs[l ][fromLayer ][destination]);
                            help2 = double(Degree[vertex ][l]);
                            ratiovalue = ratiovalue - 2*LogFunction((help1 + help2)/(LL[fromLayer][l]+LL[l][fromLayer])) + 2*LogFunction(help1/(LL[fromLayer][l]+LL[l][fromLayer]) );
                       
                       
                    


                 }
        
        
        if (LL[fromLayer][fromLayer] != 0)
                 if (NeighborLayers[ fromLayer] == 1)
                 {
                    
                    for (i=0 ; i<ActualDiffComms[fromLayer] ; i++)
                    {

                           if ((NeighborSet[ fromLayer ] [0 ][ i] != azz) && (NeighborSet[ fromLayer][0 ][i] != destination))
                           {
                              
                                   help3[NeighborSet[ fromLayer ] [0 ][ i]] += (double(NeighborSet[fromLayer ][1 ][i])/(LL[fromLayer][fromLayer]));
                           }

                           if (NeighborSet[fromLayer ][0 ][i] == azz)
                              
                                   fromcount += (double(NeighborSet[fromLayer][1 ][i])/(LL[fromLayer][fromLayer]));
                           if (NeighborSet[fromLayer][0 ][i] == destination)
                                
                                    destcount += (double(NeighborSet[fromLayer][1 ][i])/(LL[fromLayer][fromLayer]));
                                    
                    }
                    
                        
                               help1 = double(CurrentCommStubs[fromLayer ][fromLayer ][ azz]);
                               help2 = double(Degree[vertex ][fromLayer]);
                               ratiovalue = ratiovalue - 2*LogFunction((help1 - help2)/(LL[fromLayer][fromLayer]) ) + 2*LogFunction(help1/(LL[fromLayer][fromLayer]) );
                        
      

                   
                      
                        
                            help1 = double(CurrentCommStubs[fromLayer ][fromLayer ][destination]);
                            help2 = double(Degree[vertex ][fromLayer]);
                            ratiovalue = ratiovalue - 2*LogFunction((help1 + help2)/(LL[fromLayer][fromLayer])) + 2*LogFunction(help1/(LL[fromLayer][fromLayer]) );
                        
                    


                 }
        
        
             for (p=0 ; p<MaxComms ; p++)
             {
                        help1 = double(CurrentEdgeMatrixNew[azz ][p]);
                        help2 = double(CurrentEdgeMatrixNew[destination ][p]);
                        //cout<<"jjj"<<endl;
                        ratiovalue = ratiovalue + (LogFunction(help1-help3[p]) - LogFunction(help1 ));
                        ratiovalue = ratiovalue + (LogFunction(help2+help3[p]) - LogFunction(help2 ));
                        
                        
                        help1 = double(CurrentEdgeMatrixNew[p][azz ]);
                        help2 = double(CurrentEdgeMatrixNew[p][destination ]);
                        //cout<<"oooo"<<endl;
                        ratiovalue = ratiovalue + (LogFunction((help1-help3[p])) - LogFunction(help1 ));
                        ratiovalue = ratiovalue + (LogFunction((help2+help3[p])) - LogFunction(help2 ));
                        
             } 
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        //cout<<"kabiri2"<<endl;
                        help1 = double(CurrentEdgeMatrixNew[azz ][destination]);
                        help2 = double(fromcount - destcount);
                        ratiovalue = ratiovalue + (LogFunction((help1 + help2)) - LogFunction(help1));
                        
                    


                        //cout<<"kabiri4"<<endl;
                        help1 = double(CurrentEdgeMatrixNew[destination][azz]);
                        help2 = double(fromcount - destcount);
                        ratiovalue = ratiovalue + (LogFunction((help1 + help2) ) - LogFunction(help1 ));
                        
                       //cout<<"kabiri7"<<endl;
                         help1 = double(CurrentEdgeMatrixNew[azz ][azz]);
                         help2 = double(SelfEdgeCounter + 2 * fromcount);
                         ratiovalue = ratiovalue + (LogFunction((help1 - help2) ) - LogFunction(help1 )) ;
                  

                       //cout<<"kabiri11"<<endl;
                         help1 = double(CurrentEdgeMatrixNew[destination ][destination]);
                         help2 = double(SelfEdgeCounter + 2 * destcount);
                         ratiovalue = ratiovalue + (LogFunction((help1 + help2) ) - LogFunction((help1) )) ;
                 
    
    return ratiovalue;
}

void ComputeNeighborSet(int vertex, int option)
{
  unsigned int i,j,l;//-Wunused,j;
  int neighbor;
  
  SelfEdgeCounter = 0;
  
  
  for(l=0 ; l < Layers; l++)
        ActualDiffComms[l] = 0;
  for(l=0 ; l < Layers; l++)
        NeighborLayers[l] = 0;

      
 
  for(l=0 ; l < Layers; l++)
        for (j=0 ; j<MaxComms ; j++)
        {
               NeighborSet[l][0][j] = j;
               NeighborSet[l][1][j] = 0;
        }
       
   for(l=0 ; l < Layers; l++)
          for (j=0 ; j<MaxComms ; j++)
          {
                TempNeighborSet[l][0][j]=j;
                TempNeighborSet[l][1][j]=0;
          }
        // NOTE SINCE A SELF-EDGE SHOWS UP TWICE IN THE ADJLIST THIS DOUBLE
        // COUNTS THESE EDGES, WE RECORD THE NUMBER OF TIMES THIS HAPPENS
        // IN A SEPARATE VARIABLE AND THEN DIVIDE BY TWO

        for (i=0 ; i<degg[vertex] ; i++)
        {
            if (vertex != AdjMatrix[vertex][ i])
            {
                if (option == 0)

                     TempNeighborSet[Types[AdjMatrix[vertex][ i]] ][ 1 ][ CurrentState[AdjMatrix[vertex][ i]]] += 1;
                if (option == 1)


                     TempNeighborSet[Types[AdjMatrix[vertex][ i]] ][1 ][BestState[AdjMatrix[vertex][ i]]] += 1;
            }
            if (vertex == AdjMatrix[vertex][ i])
                 SelfEdgeCounter += 1;
        }


        for(l=0 ; l < Layers; l++)
        {    
                ActualDiffComms[l] = 0;


                for (i=0 ; i<MaxComms ; i++)
                    if (TempNeighborSet[l][1][i] !=0)
                    {
                        NeighborLayers[l]=1;
                        NeighborSet[l][0][ActualDiffComms[l]] = TempNeighborSet[l][0][i];
                        NeighborSet[l][1][ActualDiffComms[l]] = TempNeighborSet[l][1][i];
                        ActualDiffComms[l] +=1;
                    }
        }
  return;
}

void UpdateMatrices(int vertex, int option, int azz, int destination, int fromLayer)
{
  unsigned int i,l;//-Wunused,j;
  int fromcount = 0 ;
  int destcount = 0 ;
  
  

         if (option == 0)
         {
             //cout<<"abol1"<<endl;
            CurrentCommVertices[fromLayer][azz] -= 1;
            CurrentCommVertices[fromLayer][destination] += 1;
            for(l=0 ; l < Layers; l++)
            {
               fromcount = 0 ;
               destcount = 0 ;
              if (NeighborLayers[l]==1)
              {
                 CurrentCommStubs[fromLayer][l][azz] -= Degree[vertex][l];
                 CurrentCommStubs[fromLayer][l][destination] += Degree[vertex][l];

                 for (i=0 ; i<ActualDiffComms[l] ; i++)
                 {
                      if (NeighborSet[l][0][i]!= azz && NeighborSet[l][0][i] != destination) 
                      {
                        // do update NOTE: each community mcc' gets updated once if it had edges switch out
                        // which is correct, remembering that mcc' is symmetric and we only count c < c' here

                          CurrentEdgeMatrix[fromLayer][l][azz][NeighborSet[l][0][i]] -= NeighborSet[l][1][i];
                          CurrentEdgeMatrix[l][fromLayer][NeighborSet[l][0][i]][azz] -= NeighborSet[l][1][i];

                          CurrentEdgeMatrix[fromLayer][l][destination][NeighborSet[l][0][i]] += NeighborSet[l][1][i];
                          CurrentEdgeMatrix[l][fromLayer][NeighborSet[l][0][i]][destination] += NeighborSet[l][1][i];
                      }

                      if (NeighborSet[l][0][i] == azz) 
                           fromcount = NeighborSet[l][1][i];

                      if (NeighborSet[l][0][i] == destination) 
                           destcount = NeighborSet[l][1][i];
                 }
                 if (l != fromLayer)
                 {
                             CurrentEdgeMatrix[fromLayer][l][azz][azz] -= ( fromcount);
                             CurrentEdgeMatrix[fromLayer][l][destination][destination] += ( destcount);
                             CurrentEdgeMatrix[fromLayer][l][azz][destination] -= (  destcount);
                             CurrentEdgeMatrix[fromLayer][l][destination][azz] += (fromcount );
                             CurrentEdgeMatrix[l][fromLayer][azz][azz] -= (  fromcount);
                             CurrentEdgeMatrix[l][fromLayer][destination][destination] += ( destcount);
                             CurrentEdgeMatrix[l][fromLayer][destination][azz] -= ( destcount);
                             CurrentEdgeMatrix[l][fromLayer][azz][destination] += (fromcount );
                 }
                 else
                 {
                             CurrentEdgeMatrix[fromLayer][l][azz][azz] -= (SelfEdgeCounter + 2*fromcount);
                             CurrentEdgeMatrix[fromLayer][l][destination][destination] += (SelfEdgeCounter + 2*destcount);
                             CurrentEdgeMatrix[fromLayer][l][azz][destination] += (fromcount-  destcount);
                             CurrentEdgeMatrix[fromLayer][l][destination][azz] += (fromcount-  destcount);
                 }          
             
              }
            }
         }
         if (option == 1)
         {
             //cout<<"abol2"<<endl;
             BestCommVertices[fromLayer][azz] -= 1;
             BestCommVertices[fromLayer][destination] += 1;
             for(l=0 ; l < Layers; l++)
             {
                 fromcount = 0 ;
                 destcount = 0 ;
                if (NeighborLayers[l]==1)
                {
                   BestCommStubs[fromLayer][l][azz] -= Degree[vertex][l];
                   BestCommStubs[fromLayer][l][destination] += Degree[vertex][l];

                   for (i=0 ; i<ActualDiffComms[l] ; i++)
                   {
                         if (NeighborSet[l][0][i]!= azz &&  NeighborSet[l][0][i]!= destination) 
                         {
                             // do update NOTE: each community mcc' gets updated once if it had edges switch out
                             // which is correct, remembering that mcc' is symmetric and we only count c < c' here

                              BestEdgeMatrix[fromLayer][l][azz][NeighborSet[l][0][i]] -= NeighborSet[l][1][i];
                              BestEdgeMatrix[l][fromLayer][NeighborSet[l][0][i]][azz] -= NeighborSet[l][1][i];

                              BestEdgeMatrix[fromLayer][l][destination][NeighborSet[l][0][i]] += NeighborSet[l][1][i];
                              BestEdgeMatrix[l][fromLayer][NeighborSet[l][0][i]][destination] += NeighborSet[l][1][i];

                         }
                         if (NeighborSet[l][0][i]==azz) 
                                fromcount = NeighborSet[l][1][i];

                         if (NeighborSet[l][0][i]== destination)
                                destcount = NeighborSet[l][1][i];
                   }
                   if (l!=fromLayer)
                   {
                                BestEdgeMatrix[fromLayer][l][azz][azz] -= ( fromcount);
                                BestEdgeMatrix[fromLayer][l][destination][destination] += ( destcount);
                                BestEdgeMatrix[fromLayer][l][azz][destination] -= ( destcount);
                                BestEdgeMatrix[fromLayer][l][destination][azz] += (fromcount );
                                BestEdgeMatrix[l][fromLayer][azz][azz] -= ( fromcount);
                                BestEdgeMatrix[l][fromLayer][destination][destination] += ( destcount);
                                BestEdgeMatrix[l][fromLayer][destination][azz] -= ( destcount);
                                BestEdgeMatrix[l][fromLayer][azz][destination] += (fromcount );
                   }
                   else
                   {
                                 BestEdgeMatrix[fromLayer][l][azz][azz] -= (SelfEdgeCounter + 2*fromcount);
                                 BestEdgeMatrix[fromLayer][l][destination][destination] += (SelfEdgeCounter + 2*destcount);
                                 BestEdgeMatrix[fromLayer][l][azz][destination] += (fromcount-  destcount);
                                 BestEdgeMatrix[fromLayer][l][destination][azz] += (fromcount-  destcount);
                   }
                }
             }
         }
  return;
}

// This function returns zero if x = 0, otherwise it returns x*log(x)
double LogFunction(double x)
{
  if(x < 0)
  {
    
    cout<<"SOMETHING WRONG HAS OCCURRED STOP!"<< x <<"   is below zero:"  <<endl;
    return 0;
  }
  
  if(x == 0)
  {
    return 0;
  }
  
  return x*log(x);
}

// We do not normalize VI here.
double ComputeVI()
{
  double EntropyA;
  double EntropyB;
  double EntropyAB;
  
  EntropyA = Entropy(0); // 0 calls for best state
  EntropyB = Entropy(1); // 1 calls for true state
  EntropyAB = JointEntropy(); // does joint for best / true
  
  return 2*EntropyAB-EntropyA-EntropyB;
}

double ComputeNMI()
{
  double EntropyA;
  double EntropyB;
  double EntropyAB;
  
  EntropyA = Entropy(0);
  EntropyB = Entropy(1);
  EntropyAB = JointEntropy();
  cout<<"EntropyA ="<<EntropyA <<endl;
  cout<<"EntropyB ="<<EntropyB <<endl;
  cout<<"EntropyAB ="<<EntropyAB <<endl;
  return 2*(EntropyA+EntropyB-EntropyAB)/(EntropyA+EntropyB);
}

double Entropy(int entoption)
{
  cout<<"hosh1"<<endl;
  double Ent = 0;
  
  unsigned int i, j;//-Wunused, k;
  double *Ni;
  
  Ni = new double [MaxComms];
  
  for(i = 0; i < MaxComms; i++)
  {
    Ni[i] = 0;
  }
  
  for(j=0; j < Nodes; j++)
  {
    if(entoption == 0)
      Ni[BestState[j]]++;
    if(entoption == 1)
      Ni[TrueState[j]]++;
  }
  
  // NOTE WE RETURN THE ENTROPY IN LOG BASE 2
  for(i=0; i < MaxComms; i++)
  {
    if(Ni[i] != 0)
    {
      Ent = Ent - (Ni[i]/double(Nodes))*(log(Ni[i]/double(Nodes))/log(2));
    }
  }
  
  for(i=0; i < MaxComms; i++)
       cout<<Ni[i]<<' ';
  cout<<endl;
  delete [] Ni;
  
  return Ent;
}

// Calculates the joint entropy
double JointEntropy()
{
  unsigned int i, j, l;
  double JointEnt = 0;
  //double *Nij;
  //Nij = new double [MaxComms][MaxComms];
  double  Nij[MaxComms][MaxComms];
  
  // This rapidly calculates Nij in a simple fashion.
  for(i=0; i < MaxComms; i++)
  {
    for(j=0; j < MaxComms; j++)
    {
      Nij[i][j] = 0;
    }
  }
  
  for(l=0; l < Nodes; l++)
  {
    Nij[BestState[l]][TrueState[l]]++;
  }
  
  JointEnt = 0;
  for(i=0; i < MaxComms; i++)
  {
    for(j = 0; j < MaxComms; j++)
    {
      if(Nij[i][j] != 0)
      {
	// divide by log 2 to convert to base 2.
	JointEnt = JointEnt - (Nij[i][j]/double(Nodes))*(log(Nij[i][j]/double(Nodes))/log(2));
      }
    }
  }
  cout<<"hosh2"<<endl;
  for(i=0; i < MaxComms; i++)
  {
      for(j=0; j < MaxComms; j++)
         cout<<Nij[i][j]<<' ';
      cout<<endl;
  }
  cout<<endl;
  //delete [] Nij;
  return JointEnt;
}

/***** CPP ONLY *****/
void GetTheNetwork(string fileName1,string fileName2,string fileName3)
{
    
int i,j,row,col;
int deg=0;
int temp[Nodes];


std::ifstream file1(fileName1);
if (file1.is_open())
    for(row = 0; row < Nodes; row++)
    {
        std::string line1;
        std::getline(file1, line1);
        if ( !file1.good() )
            break;

        std::stringstream iss1(line1);
        deg=0;
        for (col = 0; col < Nodes; col++)
            temp[col]=0;
        for (col = 0; col < Nodes; col++)
        {
            std::string val1;
            std::getline(iss1, val1, ',');
            

            std::stringstream convertor(val1);
            convertor >> temp[col];
            if(temp[col]!=0)
                deg++;
        }
        degg[row]=deg;
        AdjMatrix[row] = new int [degg[row]];
        deg=0;
        for (col = 0; col < Nodes; col++)
            if(temp[col]!=0)
            {
               AdjMatrix[row][deg]=col;
               deg++;
            }
        if(deg!=degg[row])
            cout << "Unable to open store AdjMatrix ";
    }
else 
   cout << "Unable to open file";




std::ifstream file2(fileName2);
if (file2.is_open())
    for(row = 0; row < Nodes; row++)
    {
        std::string line2;
        std::getline(file2, line2);
        if ( !file2.good() )
            break;

        std::stringstream iss2(line2);

        for (col = 0; col < 1; col++)
        {
            std::string val2;
            std::getline(iss2, val2, '\n');
            //if ( !iss2.good() )
                //break;

            std::stringstream convertor(val2);
            convertor >> Comms[row];
        }
    }
else 
   cout << "Unable to open file";


std::ifstream file3(fileName3);
if (file3.is_open())
    for(row = 0; row < Nodes; row++)
    {
        std::string line3;
        std::getline(file3, line3);
        if ( !file3.good() )
            break;

        std::stringstream iss3(line3);

        for (col = 0; col < 1; col++)
        {
            std::string val3;
            std::getline(iss3, val3, '\n');
            //if ( !iss3.good() )
                //break;

            std::stringstream convertor(val3);
            convertor >> Types[row];
        }
    }
else 
   cout << "Unable to open file";



    //Initialize();
    return;
}

void PrintResults()
{
    int i;
    for (i=0 ; i< Nodes ; i++) 
        
        cout << i << "    " << BestState[i] << endl;

    if (TrueCommsAvailable == 1)
    {
        cout << "correct communities" << endl;;
        for (i=0 ; i< Nodes ; i++) 
            
            cout << i << "    " << TrueState[i] << endl;
    }

    cout <<"VI Value: " << VIValue << " NMIValue: " << NMIValue << " (Prop.) Log-Likelihood: " << HighestScore << endl ;



    
    
    return;
}





