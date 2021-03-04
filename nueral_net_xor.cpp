//implementation of nueral net xor function
#include <iostream>
#include<bits/stdc++.h>

using namespace std;

void read_data ();
//function to read the data from a text file

void write_data ();
//function to print the data obtained from the text file

float sigmoid_function ( float x );
//function to calculate the sigmmoid function of a value

float error_function ( vector < float > calculated_outcome );
//function to calculate the sum of squared errors

float weight_update_external_node ( vector < float > cal_outcome, float w_new, vector < float > yn );
//function to update the weights of the external node

float weight_update_internal_node ( vector < float > cal_outcome, float w_new , vector < float > yi, vector <float> yf, float w1, float w2 );
//function to update the values of the internal node

float check_h2 ( float val );
//function to check the h2 value

float check_h1 ( float val );
//function to check the h1 value

void feed_forward_network (int count );
//this function is the actual algorithm the output values of each of the node is calculated and weight update function is call to update the weights
//whenever necessary

void predict(int x1 , int x2);
//function to predict the output when two input values are given with the help of weights calculated earlier

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
      x1.push_back (x);
      //reading the values of vector x1
      cin>>x;
      x2.push_back (x);
      //reading the values of vector x2
      cin>>x;
      y.push_back (x);
      //reading the values of vector y
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
//function to calculate the sigmmoid function of a value
{
    float res;

    res = 1 / ( 1 + exp( - x ) );
    //1 divided by 1 plus e power minus x

    return res;
}

float error_function ( vector < float > calculated_outcome )
//function to calculate the sum of squared errors
{
    float error = 0.0;

    for( int i = 0; i < 4; i++ )
    {
        error += ( calculated_outcome[i] - y[i] ) *  ( calculated_outcome[i] - y[i] );
    }
    //the formula for the sum of squared errors is 1 divivded by 2 multiplied by the summation of the squares of the difference between the calculated 
    //outcome and the actual outcome

    error /= 2;
    return error;
    //the calculated value of error is returned
}

float weight_update_external_node ( vector < float > cal_outcome, float w_new, vector < float > yn )
//function to update the weights of the external node
{
   float z ,t, derivative;
   for( int i = 0; i < cal_outcome.size(); i++)
   {
       z = cal_outcome[i];
       t = y[i] - cal_outcome[i];

       derivative = ( z - t ) * z * (1 -z) * yn[i];
       //the partial derivative dE/dW is reduced to get the above equation

       w_new -= learning_rate * derivative;
       //the amount of change in the weight is the product of the learning rate and the derivative
   }
   learning_rate -= 0.005;
   //initiallly we assumed the learning rate as 0.6 and we gradually decrease it rather than keeping it very high or low
   
   return w_new;
   //the updated value of the weight is returned
}

float weight_update_internal_node ( vector < float > cal_outcome, float w_new , vector < float > yi, vector <float> yf, float w1, float w2 )
//function to update the values of the internal node
{
   float z ,t, derivative, dz = 0;
   for( int i = 0; i < cal_outcome.size(); i++)
   {
       z = cal_outcome[i];
       t = y[i] - cal_outcome[i];

       dz = ( z - t ) * z * (1 -z) * w1;
       dz += ( z - t ) * z * (1 -z) * w2;
       //summation of dz value is calculated

       derivative = dz * ( yi[i] * (1 - yi[i] ) * yf[i] );
       //the above equation is the partial derivative of the dE/dW

       if(t>=0)
       {
            w_new += learning_rate * derivative;
       }
       else
       {
            w_new -= learning_rate * derivative;
       }
       //the weight is updated based on the difference between the actual value and calculated value
   }
    learning_rate -= 0.005;
   //initiallly we assumed the learning rate as 0.6 and we gradually decrease it rather than keeping it very high or low
   
   return w_new;
   //the updated value of the weight is returned
}

float check_h2 ( float val )
//function to check the h2 value
{
    if (val > 0.5) 
    {
      return 1;
    } 
    return 0;
    //if the value is greater than 0.5 1 is returned or else 0 is returned
}

float check_h1 ( float val )
//function to check the h1 value
{
    if(val > 0)
    {
        return 1;
    }
    return 0;
    //if the value is greater than zero 1 is returned or else 0 is returned
}

void feed_forward_network (int count )
//this function is the actual algorithm the output values of each of the node is calculated and weight update function is call to update the weights
//whenever necessary
{
    int value = 0;
    w31 = 1, w41 = 1, w42 = 1, w23 = 1, w53 = 0.5, w54 = 0.5;
    //initial assignment of the weights in the above case it is random

    vector < float > y_n3;
    //output from the node3

    vector < float > y_n4;
    //output from the node4

    vector < float > y_n5;
    //output from the node5
   
   calculate: 
   // cout << w31 <<" "<<w41<<" "<<w42<<" "<<w23<<" "<<w53<<" "<<w54<<"\n";
  
   for( int i = 0; i < 4; i++ )
   {
       float val;
       val = ( -x1[i] * w31 ) + ( x2[i] * w23 );
       val = check_h1( val );
       y_n3.push_back( val );
       //to calculate the output from the node 3 and the value is pushed to the y_n3 vector
   }
   
   for(int i = 0; i < 4; i++)
   {
       float val;
       val = ( x1[i] * w41 ) + ( -x2[i] * w42 );
       val = check_h1( val );
       y_n4.push_back( val ) ;
      //to calculate the output from the node 4 and the value is pushed to the y_n4 vector
   }
   
   for(int i = 0; i < 4; i++)
   {
       float val;
       val = ( y_n3[i] * w53 ) + ( y_n4[i] * w54 );
       val = check_h2( val);
       y_n5.push_back( val );
       //to calculate the output from the node 5 and the value is pushed to the y_n5 vector
   }
   

   if ( error_function ( y_n5 ) > 0.2 )
   //if the error rate is greater than 0.2 then the weights are updated
   {
       w53 = weight_update_external_node ( y_n5, w53, y_n3 );
       w54 = weight_update_external_node ( y_n5, w54, y_n4 );
       //weight updation for the external nodes

       w31 = weight_update_internal_node ( y_n3, w31, y_n3, y_n5, w31, w41 );
       w41 = weight_update_internal_node ( y_n4, w41, y_n4, y_n5, w41, w42 );
       w23 = weight_update_internal_node ( y_n3, w23, y_n3, y_n5, w23, w31 );
       w42 = weight_update_internal_node ( y_n4, w42, y_n4, y_n5, w42, w23 );
       //weight_updation for the internal nodes
       
      // cout<<w53<<" "<<w54<<" "<<w31<<" "<<w41<<" "<<w23<<" "<<w42<<"\n";

       if( value < count)
       {
           value++;
           goto calculate;
           //goto the label named calculate

       }
   }

}
void predict(int x1 , int x2)
//function to predict the output when two input values are given with the help of weights calculated earlier
{
    float ans, y1, y2, y3;

    y1 = ( -x1 * w31 ) + ( x2 * w23 );
    y1 = check_h1 ( y1 );
    //output from node3 with the help of weights w31 and w23
    
    y2 = ( x1 * w41 ) + ( -x2 * w42 );
    y2 = check_h1 ( y2 );
    //output from node4 with the help of weights w41 and w42
    
    y3 = ( sigmoid_function ( y1 ) * w53 ) + ( sigmoid_function ( y2 ) * w54 );
   //output from node5 with the help of weights w53 , w54 and earlier calculated values
    
    cout<< "x1 = "<<x1 <<" x2 = "<<x2<< " answer is "<<check_h2 ( y3 ) <<"\n";
}
int main()
{
    read_data ();
    //function call to read the data from a text file

    //write_data ();
    //function call to print the data obtained from the text file

    feed_forward_network ( 5 );
    //this function is the actual algorithm the output values of each of the node is calculated and weight update function is call to update the weights
    //whenever necessary

    predict (0, 0);
    predict (0, 1);
    predict (1, 0);
    predict (1, 1);
    //function call to predict the output when two input values are given with the help of weights calculated earlier

    return 0;
}