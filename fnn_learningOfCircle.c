#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define dots_per_cycle 20
#define cycle_num 1
#define total_dots dots_per_cycle * cycle_num//104
#define learning_rate 0.001
#define epoch 10000

void makeErrorGraph(int n,double error[n]);
void makeCircleGraph(int row,int col,double array[][col]);
void normalization_reverse(int row,int col,double out_state[][col],double v[2]);
void normalization(int row,int col,double in_state[][col],double v[2]);
void subVecotr(int row,double bias[row],double delta_bias[row]);
void subMatrix(int row,int col,double weight[][col],double delta_weight[][col]);
void matrixtoVector(int row,int col,double array[][col],double vector[col]);
void delta_o_inter_cal(int row,int col,double out_state[][col],double teach_state[][col],double delta_o_inter[][col]);
void delta_h_inter_cal(int row,int col,double hidden_state[][col],double array[][col],double delta_h_inter[][col]);
double get_error(int row,int col,double teach_state[][col],double out_state[][col]);
void transpose(int row,int col,double array[][col],double tran_array[][row]);
void dotproduct(int ans_row,int ans_col,int com,double a[][com],double b[][ans_col],double ans_array[][ans_col]);
void sumMatrix(int row,int col,double a[][col],double b[col],double ans_array[][col]);
double randomnumber(void);
void makeMatrix(int row,int col,double array[][col]);
void makeVector(int row,double vector[row]);
double make_data_cos(float,int);
double make_data_sin(float,int);
void rollMatrix(double array[][2],double sub_array[][2]);

int main(){
  int i,j,n = 0;
  float r = 0.0;
  //初期化
  int length;
  int io_state_size = 2;
  int h_state_size = 20;

  double in_state[total_dots][2];
  double teach_state[total_dots][2];

  double error[epoch];

  //make_dataを実行し、in_stateに代入
  for(i=0;i<total_dots;i++){
    in_state[i][0] = make_data_cos(r = 5,i);
    in_state[i][1] = make_data_sin(r = 5,i);
  }
  //正規化
  double v[2];
  normalization(total_dots,2,in_state,v);

  length = total_dots;//104
  double hidden_state[length][h_state_size];//(104,10)
  double out_state[length][io_state_size];//(104,2)
  //target data作成
  rollMatrix(in_state, teach_state);

  double h_inter_state[length][h_state_size];//(104,10)
  double o_inter_state[length][io_state_size];

  double weight_hi[h_state_size][io_state_size];//(10,2)
  double weight_oh[io_state_size][h_state_size];//(2,10)
  for(i=0;i<h_state_size;i++){
    for(j=0;j<io_state_size;j++){
      weight_hi[i][j] = randomnumber();
    }
  }
  for(i=0;i<io_state_size;i++){
    for(j=0;j<h_state_size;j++){
      weight_oh[i][j] = randomnumber();
    }
  }
  double bias_h[h_state_size];//10

  makeVector(h_state_size,bias_h);
  for(i=0;i<h_state_size;i++){
    bias_h[i] = randomnumber();
    printf("g%f", bias_h[i]);
  }

  double bias_o[io_state_size];//2
  makeVector(io_state_size,bias_o);
  for(i=0;i<io_state_size;i++){
    bias_o[i] = randomnumber();
  }

  double delta_weight_hi[h_state_size][io_state_size];//(10,2)
  makeMatrix(h_state_size,io_state_size,delta_weight_hi);
  double delta_weight_oh[io_state_size][h_state_size];//(2,19)
  makeMatrix(io_state_size,h_state_size,delta_weight_oh);
  double delta_bias_h[h_state_size];//(2)
  makeVector(h_state_size,delta_bias_h);
  double delta_bias_o[io_state_size];//(2)
  makeVector(io_state_size,delta_bias_o);

  double tran_weight_hi[io_state_size][h_state_size];//(2,10)
  double trans_weight_oh[h_state_size][io_state_size];//(2,10)

  double array_cal1[length][h_state_size];

  double trans_delta_h_inter[h_state_size][length];
  makeMatrix(h_state_size,length,trans_delta_h_inter);
  double trans_delta_o_inter[io_state_size][length];
  makeMatrix(io_state_size,length,trans_delta_o_inter);

  double delta_o_inter[length][io_state_size];
  makeMatrix(length,io_state_size,delta_o_inter);
  double delta_h_inter[length][h_state_size];
  makeMatrix(length,h_state_size,delta_h_inter);

  //traning開始
  for(n=0;n<epoch;n++){
  //foraward_hidden
  transpose(h_state_size,io_state_size,weight_hi,tran_weight_hi);
  dotproduct(length,h_state_size,2,in_state,tran_weight_hi,h_inter_state);
  sumMatrix(length,h_state_size,h_inter_state,bias_h,h_inter_state);
  for(i=0;i<length;i++){
    for(j=0;j<h_state_size;j++){
      hidden_state[i][j] = tanh(h_inter_state[i][j]);
    }
  }
  //forward_output
  transpose(io_state_size,h_state_size,weight_oh,trans_weight_oh);
  dotproduct(length,io_state_size,h_state_size,hidden_state,trans_weight_oh,o_inter_state);
  sumMatrix(length,io_state_size,o_inter_state,bias_o,o_inter_state);
  for(i=0;i<length;i++){
    for(j=0;j<io_state_size;j++){
      out_state[i][j] = tanh(o_inter_state[i][j]);
    }
  }
  //get_error
  error[n] = get_error(length,io_state_size,teach_state,out_state);
  printf("epoch:%d =%f\n", n,error[n]);
  //backward
  delta_o_inter_cal(length,io_state_size,out_state,teach_state,delta_o_inter);
  dotproduct(length,h_state_size,2,delta_o_inter,weight_oh,array_cal1);
  delta_h_inter_cal(length,h_state_size,hidden_state,array_cal1,delta_h_inter);
  transpose(length,h_state_size,delta_h_inter,trans_delta_h_inter);
  transpose(length,io_state_size,delta_o_inter,trans_delta_o_inter);
  dotproduct(h_state_size,io_state_size,length,trans_delta_h_inter,in_state,delta_weight_hi);
  dotproduct(io_state_size,h_state_size,length,trans_delta_o_inter,hidden_state,delta_weight_oh);
  matrixtoVector(length,h_state_size,delta_h_inter,delta_bias_h);
  matrixtoVector(length,io_state_size,delta_o_inter,delta_bias_o);

  //update_parameters
  subMatrix(h_state_size,io_state_size,weight_hi,delta_weight_hi);
  subMatrix(io_state_size,h_state_size,weight_oh,delta_weight_oh);
  subVecotr(h_state_size,bias_h,delta_bias_h);
  subVecotr(io_state_size,bias_o,delta_bias_o);
}//epoch終了地点
  normalization_reverse(length,io_state_size,out_state,v);

  makeCircleGraph(length,io_state_size,out_state);
  makeErrorGraph(epoch,error);

  return 0;
}
void delta_o_inter_cal(int row,int col,double out_state[][col],double teach_state[][col],double delta_o_inter[][col]){
  int i,j;
  for(i=0;i<row;i++){
    for(j=0;j<col;j++){
      delta_o_inter[i][j] = (1.0 - pow(out_state[i][j],2.0)) * (out_state[i][j] - teach_state[i][j]);
    }
  }
}
void delta_h_inter_cal(int row,int col,double hidden_state[][col],double array[][col],double delta_h_inter[][col]){
  int i,j;
  for(i=0;i<row;i++){
    for(j=0;j<col;j++){
      delta_h_inter[i][j] = (1.0 - pow(hidden_state[i][j],2.0)) * array[i][j];
    }
  }
}
void matrixtoVector(int row,int col,double array[][col],double vector[col]){
  int i,j;
  double store = 0.0;
  for(i=0;i<col;i++){
    store = 0.0;
    for(j=0;j<row;j++){
      store += array[j][i];
    }
    vector[i] = store;
  }

}
void subMatrix(int row,int col,double weight[][col],double delta_weight[][col]){
  int i,j;
  for(i=0;i<row;i++){
    for(j=0;j<col;j++){
      weight[i][j] = weight[i][j] - (learning_rate * delta_weight[i][j]);
    }
  }
}
void subVecotr(int row,double bias[row],double delta_bias[row]){
  int i;
  for(i=0;i<row;i++){
    bias[i] = bias[i] - (learning_rate * delta_bias[i]);
  }
}

double get_error(int row,int col,double teach_state[][col],double out_state[][col]){
  int i,j;
  double error = 0;
  for(i=0;i<row;i++){
    for(j=0;j<col;j++){
      error += pow((teach_state[i][j] - out_state[i][j]),2.0);
    }
  }
  return error * 0.5;
}

//matrixの足し算
void sumMatrix(int row,int col,double a[][col],double b[col],double ans_array[][col]){
  int i,j;

  for(i=0;i<row;i++){
    for(j=0;j<col;j++){
      ans_array[i][j] = a[i][j] + b[j];
    }
  }
}
void transpose(int row,int col,double array[][col],double trans_array[][row]){
  int i,j;
  for(i=0;i<row;i++){
    for(j=0;j<col;j++){
        trans_array[j][i] = array[i][j];
    }
  }
}
void dotproduct(int ans_row,int ans_col,int com,double a[ans_row][com],double b[com][ans_col],double ans_array[ans_row][ans_col]){//a(20,2) b(2,10)
  int i,j,k;
  double store = 0.0;
  for(i=0;i<ans_row;i++){
    for(j=0;j<ans_col;j++){
      store = 0.0;
      for(k=0;k<com;k++){
        //ans_array[i][j] += a[i][k] * b[k][j];
        store += a[i][k] * b[k][j];
      }
      ans_array[i][j] = store;
    }
  }


}

double make_data_cos(float r,int i){
  return r * cos(i * (2 * M_PI / dots_per_cycle));
}

double make_data_sin(float r,int i){
  return r * sin(i * (2 * M_PI / dots_per_cycle));
}

//指定した行と列で0行列を生成する関数
void makeMatrix(int row, int col,double array[][col]){
  int i,j;
  for(i = 0;i < row; i++){
    for(j = 0; j < col; j++){
      array[i][j] = 0.0;
    }
  }
}
//指定したベクトル作成する関数
void makeVector(int row,double vector[row]){
  int i;
  for(i=0;i<row;i++){
    vector[i] = 0;
  }

}
void rollMatrix(double array[][2], double sub_array[][2]){
  int k;
  sub_array[0][0] = array[total_dots-1][0];
  sub_array[0][1] = array[total_dots-1][1];
  for(k = 1;k < total_dots;k++){
    sub_array[k][0] = array[k-1][0];
    sub_array[k][1] = array[k-1][1];
  }
}
//ランダム値を生成する関数
double randomnumber(void){
  double rng;
  double rng_minus;
  int bina;
  //修正
  rng = ((double)rand())/(double)RAND_MAX; //型丸め問題回避（int→double）の簡単な方法
  rng_minus = ((double)rand())/(double)RAND_MAX - 1.0;

  bina = rand()%(1-0+1);
  if(bina == 0){
     rng = rng_minus;
  }
  return rng; //0.1 * rng;
}
void makeCircleGraph(int row,int col,double array[][col]){
  int i;
  FILE *fp;
  fp = fopen("circle.csv","w");
  if(fp==NULL){
    printf("Can't open file.csv\n");
    exit(1);
  }
  for(i=0;i<row;i++){
    fprintf(fp, "%f,%f\n", array[i][0],array[i][1]);
  }
  fclose(fp);
}
void makeErrorGraph(int n,double error[n]){
  int i;
  FILE *fp;
  fp = fopen("error.csv","w");
  if(fp==NULL){
    printf("Can't open file.csv\n");
    exit(1);
  }
  for(i=0;i<n;i++){
    fprintf(fp, "%f\n", error[i]);
  }
  fclose(fp);
}
void normalization_reverse(int row,int col,double out_state[][col],double v[2]){
  int i,j;
  for(i=0;i<row;i++){
    for(j=0;j<col;j++){
      out_state[i][j] = (out_state[i][j] - (-1.0)) * (v[1] - v[0]) / (1.0 - (-1.0)) + v[0];
    }
  }
}
void normalization(int row,int col,double in_state[][col],double v[2]){
  int i,j;
  double vmax = 0.0,vmin = 0.0;
  for(i=0;i<row;i++){
    for(j=0;j<col;j++){
      if(in_state[i][j]<vmin){
        vmin = in_state[i][j];
      }
      if(in_state[i][j]>vmax){
        vmax = in_state[i][j];
      }
    }
  }
  v[0] = vmin;
  v[1] = vmax;
  for(i=0;i<row;i++){
    for(j=0;j<col;j++){
      in_state[i][j] = (in_state[i][j] - vmin) / (vmax - vmin) * (1.0 - (-1.0)) + (-1.0);
    }
  }
}
