#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

typedef struct ParamType{

    float InToFC[32][25];
    float FCBias[32];
    float SCFromFC[64][32][25];
    float SCBias[64];
    float NeFromSC[512][64][16];
    float NeBias[512];
    float OutFromNe[10][521];
    float OutBias[10];

}Parameter;

typedef struct ParamQType{

    int8_t  Conv1Weight[32][25];
    int     Conv1Bias[32];
    int8_t  Conv2Weight[64][32][25];
    int     Conv2Bias[64];
    int8_t  FC1Weight[512][64][16];
    int     FC1Bias[512];
    int8_t  FC2Weight[10][521];
    int     FC2Bias[10];

}ParamQ;

typedef struct ScaleType{
    float Conv1Input;
    float Conv1Weight;
    float Conv2Input;
    float Conv2Weight;
    float FC1Input;
    float FC1Weight;
    float FC2Input;
    float FC2Weight;
}ScaleFactor;

typedef struct PicType {
    float NUM[28][28];
}Picture;

void readweight(FILE *file, float *weight)
{
    fscanf(file,"%f",weight);
}

float Abs(float a) {
    return a > 0.0 ? a : (0-a);
}


float Max4(float a,float b, float c,float d) {
    float temp0,temp1;
    temp0 = a > b ? a : b;
    temp1 = c > d ? c : d;
    return (temp0 > temp1 ? temp0 : temp1);
}

int Max4I(int a,int b, int c,int d) {
    int temp0,temp1;
    temp0 = a > b ? a : b;
    temp1 = c > d ? c : d;
    return (temp0 > temp1 ? temp0 : temp1);
}

void ReadParam(Parameter * Par) {

    float weight_f;

    FILE *fi;

    fi = fopen("conv1_weight","r");
    for(int i = 0;i < 5;i++) {
        for(int j = 0;j < 5;j++) {
            for(int k = 0;k < 32;k++) {
                readweight(fi,&weight_f);
                Par->InToFC[k][i*5+j] = weight_f;
            }
        }
    }
    fclose(fi);

    fi = fopen("conv2_weight","r");
    for(int i = 0;i < 5;i++) {
        for(int j = 0;j < 5;j++) {
            for(int k = 0;k < 32;k++) {
                for(int m = 0;m < 64;m++) {
                    readweight(fi,&weight_f);
                    Par->SCFromFC[m][k][i*5+j] = weight_f;
                }
            }   
        }
    }
    fclose(fi);

    fi = fopen("fc1_weight","r");
    for(int i = 0;i < 4;i++) {
        for(int j = 0;j < 4;j++) {
            for(int k = 0;k < 64;k++) {
                for(int m = 0;m < 512;m++) {
                    readweight(fi,&weight_f);
                    Par->NeFromSC[m][k][i*4+j] = weight_f;
                }
            }   
        }
    }
    fclose(fi);

    fi = fopen("fc2_weight","r");
    for(int k = 0;k < 512;k++) {
        for(int m = 0;m < 10;m++) {
            readweight(fi,&weight_f);
            Par->OutFromNe[m][k] = weight_f;
        }
    } 
    fclose(fi);

    fi = fopen("conv1_bias","r");
    for(int k = 0;k < 32;k++) {
        readweight(fi,&weight_f);
        Par->FCBias[k] = weight_f;
    } 
    fclose(fi);

    fi = fopen("conv2_bias","r");
    for(int k = 0;k < 64;k++) {
        readweight(fi,&weight_f);
        Par->SCBias[k] = weight_f;
    } 
    fclose(fi);

    fi = fopen("fc1_bias","r");
    for(int k = 0;k < 512;k++) {
        readweight(fi,&weight_f);
        Par->NeBias[k] = weight_f;
    } 
    fclose(fi);

    fi = fopen("fc2_bias","r");
    for(int k = 0;k < 10;k++) {
        readweight(fi,&weight_f);
        Par->OutBias[k] = weight_f;
    } 
    fclose(fi);

    return ;
}

void ReadPic(FILE *fn, Picture *Pic) {

    int p;

    for(int i = 0;i < 28;i++) {
        for(int j = 0;j < 28;j++) {
            fscanf(fn,"%d",&p);
            Pic->NUM[i][j] = (float)p;
        }
    }
    fclose(fn);

    return ;
}

void Quantization(Parameter *Par, Picture *Pic, ScaleFactor *Factor, ParamQ *ParQ) {

    //中间结果声明
    float Conv1Out[32][24][24] = {0.0};
    float Layer1Out[32][12][12] = {0.0};
    float Conv2Out[64][8][8] = {0.0};
    float Layer2Out[64][4][4] = {0.0};
    float Ner[512] = {0.0};
    float Prob[10] = {0.0};

    int m,i,j,k,x,y,n;
    float max = 0.0;

    //Conv1Input Quanti
    max = 0.0;
    for(i = 0;i < 28;i++) {
        for(j = 0;j < 28;j++) {
            if(Abs(Pic->NUM[i][j]) > max) {
                max = Abs(Pic->NUM[i][j]);
            }
        }
    }
    Factor->Conv1Input = 127.0 / max;

    //Conv1Weight Quanti
    max = 0.0;
    for(i = 0;i < 32;i++) {
        for(j = 0;j < 25;j++) {
            if(Abs(Par->InToFC[i][j]) > max) {
                max = Abs(Par->InToFC[i][j]);
            }
        }
        if(Abs(Par->FCBias[i]) > max) {
            max = Abs(Par->FCBias[i]);
        }
    }
    Factor->Conv1Weight = 127.0 / max;
    for(i = 0;i < 32;i++) {
        for(j = 0;j < 25;j++) {
            ParQ->Conv1Weight[i][j] = (int8_t)(Par->InToFC[i][j] * Factor->Conv1Weight);
        }
        ParQ->Conv1Bias[i] = (int)(Par->FCBias[i] * Factor->Conv1Weight * Factor->Conv1Input);
    }

    //第一层
    for(i = 0;i < 32;i++) {
        //卷积
        for(j = 0;j < 24;j++) {
            for(k = 0;k < 24;k++) {
                for(x = 0;x < 5;x++) {
                    for(y = 0;y < 5;y++) {
                        Conv1Out[i][j][k] += Pic->NUM[j+x][k+y] * Par->InToFC[i][x*5+y];
                    }
                }
            }
        }
        //偏置激活
        for(j = 0;j < 24;j++) {
            for(k = 0;k < 24;k++) {
                Conv1Out[i][j][k] += Par->FCBias[i];
                Conv1Out[i][j][k] = Conv1Out[i][j][k] > 0 ? Conv1Out[i][j][k] : 0.0;
            }
        }
        //池化
        for(j = 0;j < 12;j++) {
            for(k = 0;k < 12;k++) {
                Layer1Out[i][j][k] = Max4(Conv1Out[i][j*2][k*2],Conv1Out[i][j*2+1][k*2],Conv1Out[i][j*2][k*2+1],Conv1Out[i][j*2+1][k*2+1]);
            }
        }
    }

    //Conv2Input Quanti
    max = 0.0;
    for(i = 0;i < 32;i++) {
        for(j = 0;j < 12;j++) {
            for(k = 0;k < 12;k++) {
                if(Abs(Layer1Out[i][j][k]) > max) {
                    max = Abs(Layer1Out[i][j][k]);
                }
            }
        }
    }
    Factor->Conv2Input = 127.0 / max;

    //Conv2Weight Quanti
    max = 0.0;
    for(i = 0;i < 64;i++) {
        for(j = 0;j < 32;j++) {
            for(k = 0;k < 25;k++) {
                if(Abs(Par->SCFromFC[i][j][k]) > max) {
                    max = Abs(Par->SCFromFC[i][j][k]);
                }
            }
        }
        if(Abs(Par->SCBias[i]) > max) {
            max = Abs(Par->SCBias[i]);
        }
    }
    Factor->Conv2Weight = 127.0 / max;
    for(i = 0;i < 64;i++) {
        for(j = 0;j < 32;j++) {
            for(k = 0;k < 25;k++) {
                ParQ->Conv2Weight[i][j][k] = (int8_t)(Par->SCFromFC[i][j][k] * Factor->Conv2Weight);
            }
        }
        ParQ->Conv2Bias[i] = (int)(Par->SCBias[i] * Factor->Conv2Weight * Factor->Conv2Input);
    }

    //第二层
    for(m = 0;m < 64;m++) {
        //卷积
        for(i = 0;i < 32;i++) {
            for(j = 0;j < 8;j++) {
                for(k = 0;k < 8;k++) {
                    for(x = 0;x < 5;x++) {
                        for(y = 0;y < 5;y++) {
                            Conv2Out[m][j][k] += Layer1Out[i][j+x][k+y] * Par->SCFromFC[m][i][x*5+y];
                        }
                    }
                }
            }
        }
        //偏置激活
        for(j = 0;j < 8;j++) {
            for(k = 0;k < 8;k++) {
                Conv2Out[m][j][k] += Par->SCBias[m];
                Conv2Out[m][j][k] = Conv2Out[m][j][k] > 0 ? Conv2Out[m][j][k] : 0.0;
            }
        }
        //池化
        for(j = 0;j < 4;j++) {
            for(k = 0;k < 4;k++) {
                Layer2Out[m][j][k] = Max4(Conv2Out[m][j*2][k*2],Conv2Out[m][j*2+1][k*2],Conv2Out[m][j*2][k*2+1],Conv2Out[m][j*2+1][k*2+1]);
            }
        }
    }

    //FC1Input Quanti
    max = 0.0;
    for(i = 0;i < 64;i++) {
        for(j = 0;j < 4;j++) {
            for(k = 0;k < 4;k++) {
                if(Abs(Layer2Out[i][j][k]) > max) {
                    max = Abs(Layer2Out[i][j][k]);
                }
            }
        }
    }
    Factor->FC1Input = 127.0 / max;

    //FC1Weight Quanti
    max = 0.0;
    for(i = 0;i < 512;i++) {
        for(j = 0;j < 64;j++) {
            for(k = 0;k < 16;k++) {
                if(Abs(Par->NeFromSC[i][j][k]) > max) {
                    max = Abs(Par->NeFromSC[i][j][k]);
                }
            }
        }
        if(Abs(Par->NeBias[i]) > max) {
            max = Abs(Par->NeBias[i]);
        }
    }
    Factor->FC1Weight = 127.0 / max;
    for(i = 0;i < 512;i++) {
        for(j = 0;j < 64;j++) {
            for(k = 0;k < 16;k++) {
                ParQ->FC1Weight[i][j][k] = (int8_t)(Par->NeFromSC[i][j][k] * Factor->FC1Weight);
            }
        }
        ParQ->FC1Bias[i] = (int)(Par->NeBias[i] * Factor->FC1Weight * Factor->FC1Input);
    }

    //全连接第一层
    for(i = 0;i < 512;i++) {
        //全连接
        for(j = 0;j < 64;j++) {
            for(x = 0;x < 4;x++) {
                for(y = 0;y < 4;y++) {
                    Ner[i] += Layer2Out[j][x][y] * Par->NeFromSC[i][j][x*4+y];
                }
            }
        }
        //偏置激活
        Ner[i] += Par->NeBias[i];
        Ner[i] = Ner[i] > 0 ? Ner[i] : 0.0;
    }

    //FC2Input Quanti
    max = 0.0;
    for(i = 0;i < 512;i++) {
        if(Abs(Ner[i]) > max) {
            max = Abs(Ner[i]);
        }
    }
    Factor->FC2Input = 127.0 / max;

    //FC1Weight Quanti
    max = 0.0;
    for(i = 0;i < 10;i++) {
        for(j = 0;j < 512;j++) {
            if(Abs(Par->OutFromNe[i][j]) > max) {
                max = Abs(Par->OutFromNe[i][j]);
            }
        }
        if(Abs(Par->OutBias[i]) > max) {
            max = Abs(Par->OutBias[i]);
        }
    }
    Factor->FC2Weight = 127.0 / max;
    for(i = 0;i < 10;i++) {
        for(j = 0;j < 512;j++) {
            ParQ->FC2Weight[i][j] = (int8_t)(Par->OutFromNe[i][j] * Factor->FC2Weight);
        }
        ParQ->FC2Bias[i] = (int)(Par->OutBias[i] * Factor->FC2Weight * Factor->FC2Input);
    }

    //全连接第二层
    for(i = 0;i < 10;i++) {
        //全连接
        for(j = 0;j < 512;j++) {
            Prob[i] += Ner[j] * Par->OutFromNe[i][j];
        }
        //偏置
        Prob[i] += Par->OutBias[i];
    }

    return;

}

int InferFloat(Parameter *Par, Picture *Pic) {

    //中间结果声明
    float Conv1Out[32][24][24] = {0.0};
    float Layer1Out[32][12][12] = {0.0};
    float Conv2Out[64][8][8] = {0.0};
    float Layer2Out[64][4][4] = {0.0};
    float Ner[512] = {0.0};
    float Prob[10] = {0.0};

    int m,i,j,k,x,y,n;
    float max = 0.0;

    //第一层
    for(i = 0;i < 32;i++) {
        //卷积
        for(j = 0;j < 24;j++) {
            for(k = 0;k < 24;k++) {
                for(x = 0;x < 5;x++) {
                    for(y = 0;y < 5;y++) {
                        Conv1Out[i][j][k] += Pic->NUM[j+x][k+y] * Par->InToFC[i][x*5+y];
                    }
                }
            }
        }
        //偏置激活
        for(j = 0;j < 24;j++) {
            for(k = 0;k < 24;k++) {
                Conv1Out[i][j][k] += Par->FCBias[i];
                Conv1Out[i][j][k] = Conv1Out[i][j][k] > 0 ? Conv1Out[i][j][k] : 0.0;
            }
        }
        //池化
        for(j = 0;j < 12;j++) {
            for(k = 0;k < 12;k++) {
                Layer1Out[i][j][k] = Max4(Conv1Out[i][j*2][k*2],Conv1Out[i][j*2+1][k*2],Conv1Out[i][j*2][k*2+1],Conv1Out[i][j*2+1][k*2+1]);
            }
        }
    }

    //第二层
    for(m = 0;m < 64;m++) {
        //卷积
        for(i = 0;i < 32;i++) {
            for(j = 0;j < 8;j++) {
                for(k = 0;k < 8;k++) {
                    for(x = 0;x < 5;x++) {
                        for(y = 0;y < 5;y++) {
                            Conv2Out[m][j][k] += Layer1Out[i][j+x][k+y] * Par->SCFromFC[m][i][x*5+y];
                        }
                    }
                }
            }
        }
        //偏置激活
        for(j = 0;j < 8;j++) {
            for(k = 0;k < 8;k++) {
                Conv2Out[m][j][k] += Par->SCBias[m];
                Conv2Out[m][j][k] = Conv2Out[m][j][k] > 0 ? Conv2Out[m][j][k] : 0.0;
            }
        }
        //池化
        for(j = 0;j < 4;j++) {
            for(k = 0;k < 4;k++) {
                Layer2Out[m][j][k] = Max4(Conv2Out[m][j*2][k*2],Conv2Out[m][j*2+1][k*2],Conv2Out[m][j*2][k*2+1],Conv2Out[m][j*2+1][k*2+1]);
            }
        }
    }

    //全连接第一层
    for(i = 0;i < 512;i++) {
        //全连接
        for(j = 0;j < 64;j++) {
            for(x = 0;x < 4;x++) {
                for(y = 0;y < 4;y++) {
                    Ner[i] += Layer2Out[j][x][y] * Par->NeFromSC[i][j][x*4+y];
                }
            }
        }
        //偏置激活
        Ner[i] += Par->NeBias[i];
        Ner[i] = Ner[i] > 0 ? Ner[i] : 0.0;
    }

    //全连接第二层
    for(i = 0;i < 10;i++) {
        //全连接
        for(j = 0;j < 512;j++) {
            Prob[i] += Ner[j] * Par->OutFromNe[i][j];
        }
        //偏置
        Prob[i] += Par->OutBias[i];
    }

    max = Prob[0];
    uint16_t ArgMax = 0;

    for(i = 0;i < 10;i++) {
        printf("%f ",Prob[i]);
        if(Prob[i] > max) {
            max = Prob[i];
            ArgMax = i;
        }
    }
    printf("\n");
    return ArgMax;

}

int InferInt(ParamQ *Par, Picture *Pic, ScaleFactor *Fac) {

    int8_t  Conv1In[28][28];

    int  Conv1Out[32][24][24] = {0};
    int  Layer1Out[32][12][12] = {0};
    float  Conv1Float[32][12][12];

    int8_t  Conv2In[32][12][12];
    int  Conv2Out[64][8][8] = {0};
    int  Layer2Out[64][4][4] = {0};
    float  Conv2Float[64][4][4];

    int8_t FC1In[64][4][4];
    int  Ner[512] = {0};
    float  FC1Float[512];

    int8_t FC2In[512];
    int  Prob[10] = {0};
    float  FC2Float[10];

    int m,i,j,k,x,y,n;
    float temp;

    //Quanti Conv1Input
    for(i = 0;i < 28;i++) {
        for(j = 0;j < 28;j++) {
            temp = Pic->NUM[i][j] * Fac->Conv1Input;
            if(temp > 127) temp = 127;
            if(temp < -128) temp = -128;
            Conv1In[i][j] = (int8_t)temp;
        }
    }
    //第一层
    for(i = 0;i < 32;i++) {
        //卷积
        for(j = 0;j < 24;j++) {
            for(k = 0;k < 24;k++) {
                for(x = 0;x < 5;x++) {
                    for(y = 0;y < 5;y++) {
                        Conv1Out[i][j][k] += (int)Conv1In[j+x][k+y] * (int)Par->Conv1Weight[i][x*5+y];
                    }
                }
            }
        }
        //偏置激活
        for(j = 0;j < 24;j++) {
            for(k = 0;k < 24;k++) {
                Conv1Out[i][j][k] += Par->Conv1Bias[i];
                Conv1Out[i][j][k] = Conv1Out[i][j][k] > 0 ? Conv1Out[i][j][k] : 0.0;
            }
        }
        //池化
        for(j = 0;j < 12;j++) {
            for(k = 0;k < 12;k++) {
                Layer1Out[i][j][k] = Max4I(Conv1Out[i][j*2][k*2],Conv1Out[i][j*2+1][k*2],Conv1Out[i][j*2][k*2+1],Conv1Out[i][j*2+1][k*2+1]);
            }
        }
    }
    for(i = 0;i < 32;i++) {
        for(j = 0;j < 12;j++) {
            for(k = 0;k < 12;k++) {
                Conv1Float[i][j][k] = (float)Layer1Out[i][j][k] / Fac->Conv1Input / Fac->Conv1Weight;
            }
        }
    }


    //Quanti Conv2
    for(i = 0;i < 32;i++) {
        for(j = 0;j < 12;j++) {
            for(k = 0;k < 12;k++) {
                temp = Conv1Float[i][j][k] * Fac->Conv2Input;
                if(temp > 127) temp = 127;
                if(temp < -128) temp = -128;
                Conv2In[i][j][k] = (int8_t)temp;
            }
        }
    }
    //第二层
    for(m = 0;m < 64;m++) {
        //卷积
        for(i = 0;i < 32;i++) {
            for(j = 0;j < 8;j++) {
                for(k = 0;k < 8;k++) {
                    for(x = 0;x < 5;x++) {
                        for(y = 0;y < 5;y++) {
                            Conv2Out[m][j][k] += (int)Conv2In[i][j+x][k+y] * (int)Par->Conv2Weight[m][i][x*5+y];
                        }
                    }
                }
            }
        }
        //偏置激活
        for(j = 0;j < 8;j++) {
            for(k = 0;k < 8;k++) {
                Conv2Out[m][j][k] += Par->Conv2Bias[m];
                Conv2Out[m][j][k] = Conv2Out[m][j][k] > 0 ? Conv2Out[m][j][k] : 0.0;
            }
        }
        //池化
        for(j = 0;j < 4;j++) {
            for(k = 0;k < 4;k++) {
                Layer2Out[m][j][k] = Max4I(Conv2Out[m][j*2][k*2],Conv2Out[m][j*2+1][k*2],Conv2Out[m][j*2][k*2+1],Conv2Out[m][j*2+1][k*2+1]);
            }
        }
    }
    for(i = 0;i < 64;i++) {
        for(j = 0;j < 4;j++) {
            for(k = 0;k < 4;k++) {
                Conv2Float[i][j][k] = (float)Layer2Out[i][j][k] / Fac->Conv2Input / Fac->Conv2Weight;
            }
        }
    }

    //Quanti FC1
    for(i = 0;i < 64;i++) {
        for(j = 0;j < 4;j++) {
            for(k = 0;k < 4;k++) {
                temp = Conv2Float[i][j][k] * Fac->FC1Input;
                if(temp > 127) temp = 127;
                if(temp < -128) temp = -128;
                FC1In[i][j][k] = (int8_t)temp;
            }
        }
    }
    //全连接第一层
    for(i = 0;i < 512;i++) {
        //全连接
        for(j = 0;j < 64;j++) {
            for(x = 0;x < 4;x++) {
                for(y = 0;y < 4;y++) {
                    Ner[i] += (int)FC1In[j][x][y] * (int)Par->FC1Weight[i][j][x*4+y];
                }
            }
        }
        //偏置激活
        Ner[i] += Par->FC1Bias[i];
        Ner[i] = Ner[i] > 0 ? Ner[i] : 0.0;
    }
    for(i = 0;i < 512;i++) {
        FC1Float[i] = (float)Ner[i] / Fac->FC1Weight / Fac->FC1Input;
    }

    //Quanti FC2
    for(i = 0;i < 512;i++) {
        temp = FC1Float[i] * Fac->FC2Input;
        if(temp > 127) temp = 127;
        if(temp < -128) temp = -128;
        FC2In[i] = (uint8_t)temp;
    }
    //全连接第二层
    for(i = 0;i < 10;i++) {
        //全连接
        for(j = 0;j < 512;j++) {
            Prob[i] += (int)FC2In[j] * (int)Par->FC2Weight[i][j];
        }
        //偏置
        Prob[i] += Par->FC2Bias[i];
    }
    for(i = 0;i < 10;i++) {
        FC2Float[i] = (float)Prob[i] / Fac->FC2Input / Fac->FC2Weight;
    }

    float max = 0.0;
    int ArgMax = 0;
    for(i = 0;i < 10;i++) {
        printf("%f ",FC2Float[i]);
        if(FC2Float[i] > max) {
            max = FC2Float[i];
            ArgMax = i;
        }
    }
    printf("\n");
    return ArgMax;

}

int main() {

    
    clock_t start,finish;
    float duration;

    FILE *pic0;
    FILE *pic1;

    pic0 = fopen("4.txt","r");
    pic1 = fopen("6.txt","r");

    Parameter Par;
    Picture Pic;
    ParamQ ParQ;
    ScaleFactor Fac;

    ReadParam(&Par);
    ReadPic(pic0, &Pic);
    Quantization(&Par,&Pic,&Fac,&ParQ);

    int result;
    ReadPic(pic1, &Pic);

    start = clock();
    result = InferFloat(&Par,&Pic);
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("elapsed time : %f\n",duration);

    start = clock();
    result = InferInt(&ParQ,&Pic,&Fac);
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("elapsed time : %f\n",duration);
    

    return 0;

}