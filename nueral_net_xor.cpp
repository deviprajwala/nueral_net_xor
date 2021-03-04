//implementation of nueral net xor function
#include <iostream>
#include<bits/stdc++.h>

using namespace std;

vector < bool > x1;
//vector to store the first input

vector < bool > x2;
//vector to store the second input

vector < bool > y;
//vector to store the outcome

float w31, w41, w32, w42, w23, w53, w54, learning_rate = 0.5;
//variables to store the weights 

void read_data ()
//function to read the data from a text file
{
  for( int i = 0; i < 4; i++)
  {
      int x;
      cin>>x;
      x1.push_back(x);
      cin>>x;
      x2.push_back(x);
      cin>>x;
      y.push_back(x);
  }
}

void write_data ()
//function to print the data obtained from the text file
{
  for( int i = 0; i < 4; i++)
  {
      cout<<x1[i]<<" "<<x2[i]<<" "<<y[i]<<"\n";
  }
}
float sigmoid_function ( float x )
{
    float res;

    res = 1 / ( 1 + exp( - x ) );
    return res;
}

float error_function ( vector < float > calculated_outcome )
{
    float error = 0.0;

    for( int i = 0; i < 4; i++ )
    {
        error += ( calculated_outcome[i] - y[i] ) *  ( calculated_outcome[i] - y[i] );
    }

    error /= 2;
    return error;
}

float weight_update_external_node ( vector < float > cal_outcome, float w_new, vector < float>yn )
{
   float z ,t, derivative;
   for( int i = 0; i < cal_outcome.size(); i++)
   {
       z = cal_outcome[i];
       t = y[i] - cal_outcome[i];

       derivative = ( z - t ) * z * (1 -z) * yn[i];

       w_new -= learning_rate * derivative;
   }
   learning_rate -= 0.005;
   return w_new;
}
float weight_update_internal_node (  vector < float > cal_outcome, float w_new , vector < float > yi, vector <float> yf, float w1, float w2 )
{
   float z ,t, derivative, dz = 0;
   for( int i = 0; i < cal_outcome.size(); i++)
   {
       z = cal_outcome[i];
       t = y[i] - cal_outcome[i];

       dz = ( z - t ) * z * (1 -z) * w1;
       dz += ( z - t ) * z * (1 -z) * w2;

       derivative = dz * ( yi[i] * (1 - yi[i] ) * yf[i] );
       if(t>=0)
       {
            w_new += learning_rate * derivative;
       }
       else
       {
            w_new -= learning_rate * derivative;
       }
   }
   
   return w_new;
}
float check (float val)
{
    float y;
    if (val >= 0.05 && val <= 0.15 )
    {
        y = 1;
    }
    else
    {
        y = 0;
    }
    return y;
}
void feed_forward_network (int count)
{
    int value = 0;
    w31 = 1, w41 = 1, w42 = 1, w23 = 1, w53 = 0.5, w54 = 0.5;
    vector < float > y_n3;
    vector < float > y_n4;
    vector < float > y_n5;

   calculate: 
  // cout << w31 <<" "<<w41<<" "<<w42<<" "<<w23<<" "<<w53<<" "<<w54<<"\n";
   for( int i = 0; i < 4; i++ )
   {
       float val;
       val = ( x1[i] * w31 ) + ( x2[i] * w23 );
       y_n3.push_back(sigmoid_function (val ) );
   }
   
   for(int i = 0; i < 4; i++)
   {
       float val;
       val = ( x1[i] * w41 ) + ( x2[i] * w42 );
       y_n4.push_back (sigmoid_function (val ) );
   }
   
   for(int i = 0; i < 4; i++)
   {
       float val;
       val = ( y_n3[i] * w53 ) + ( y_n4[i] * w54 );
       y_n5.push_back ( sigmoid_function (val ) );
   }
   for(int i = 0; i < 4; i++)
   {
      // cout<<y_n5[i]<<" ";
   }
  // cout<<"\n";
   //cout<< error_function (y_n5)<<"\n ";

   if ( error_function ( y_n5 ) > 0.4 )
   {
      // weight_update_external_node(y_n5);
       w53 = weight_update_external_node ( y_n5, w53, y_n3 );
       w54 = weight_update_external_node ( y_n5, w54, y_n4 );
       
       w31 = weight_update_internal_node ( y_n3, w31, y_n3, y_n5, w31, w41 );
       w41 = weight_update_internal_node ( y_n4, w41, y_n4, y_n5, w41, w42 );
       w23 = weight_update_internal_node ( y_n3, w23, y_n3, y_n5, w23, w31 );
       w42 = weight_update_internal_node ( y_n4, w42, y_n4, y_n5, w42, w23 );
       //weight_update_internal_node();
       
     //  cout<<w53<<" "<<w54<<" "<<w31<<" "<<w41<<" "<<w23<<" "<<w42<<"\n";
       if (value < count ) 
       {
           value++;
           goto calculate;
        
       }
   }

}
void predict(int x1 , int x2)
{


    float ans, y1, y2, y3;

    y1 = ( x1 * w31 ) + ( x2 * w23 );
    
    y2 = ( x1 * w41 ) + ( x2 * w42 );

    y3 = ( sigmoid_function(y1) * w53 ) + ( sigmoid_function(y2) * w54 );

   
    cout<<sigmoid_function (y3)<<" ";
}
int main()
{
    read_data();
    //write_data();
    feed_forward_network(9);
  //  feed_forward_network();
   // feed_forward_network();
   predict(0,0);
     predict(0,1);
       predict(1,0);
         predict(1,1);
    return 0;
}