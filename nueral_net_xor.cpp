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

float w31, w41, w32, w42, w53, w54, learning_rate = 0.5;
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
   learning_rate -= 0.05;
   return w_new;
}
float weight_update_internal_node ( float w_new )
{
   return w_new;
}
void feed_forward_network ()
{
    float w31 = 0.4, w41 = 0.45, w42 = 0.5, w23 = 0.55, w53 = 0.6, w54 = 0.65;
    vector < float > y_n3;
    vector < float > y_n4;
    vector < float > y_n5;

   calculate: 
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
       y_n4.push_back(sigmoid_function (val ) );
   }
   
   for(int i = 0; i < 4; i++)
   {
       float val;
       val = ( y_n3[i] * w53 ) + ( y_n4[i] * w54 );
       y_n5.push_back(sigmoid_function (val ) );
   }
   for(int i = 0; i < 4; i++)
   {
      // cout<<y_n5[i]<<" ";
   }
   if ( error_function ( y_n5 ) > 0.4 )
   {
      // weight_update_external_node(y_n5);
       w53 = weight_update_external_node ( y_n5, w53, y_n3 );
       w54 = weight_update_external_node ( y_n5, w54, y_n4 );
       cout<<w54;
       //weight_update_internal_node();
      // goto calculate;
   }

}
int main()
{
    read_data();
    //write_data();
    feed_forward_network();
    return 0;
}