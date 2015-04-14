/*
	B490/B659 Project 2 Skeleton Code    (2/2015)
	
	Be sure to read over the project document and this code (should you choose to use it) before 
	starting to program. 
	

	Compiling:
		A simple console command 'make' will excute the compilation instructions
		found in the Makefile bundled with this code. It will result in an executable
		named p2.

	Running:
		The executable p2 should take commands of the form:
			./p2 problem_ID input_File ouput_File additional_Arguments
	
*/


//Link to the header file
#include "CImg.h"
#include <ctime>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <fft.h>
#include <math.h>
#define TWO_PI 6.2831853071795864769252866
//Use the cimg namespace to access the functions easily
using namespace cimg_library;
using namespace std;

//Part 2 - Morphing
int ImgBorderProcess(int ij, int wh);
CImg<double> filter(CImg<double> input, CImg<double> filter);
CImg<double> gaussianFilter(CImg<double> input, double sigma);
CImg<double> morph(CImg<double> input, CImg<double> input1,double sigma);

//Part 3 - Spectrogram
CImg<double> averageGrayscale(CImg<double> input);
CImg<double> resize(CImg<double> input,int num);
CImg<double> fft_magnitude(const CImg<double> &fft_real, const CImg<double> &fft_imag);
CImg<double> remove_interference(const CImg<double> &input);
CImg<double> fft_filter(const CImg<double> &input, const CImg<double> &filter);
CImg<double> fft_defilter(const CImg<double> &input, double sigma);
CImg<double> fft_defilter_mean(CImg<double> &input, int filterSize);
CImg<double> spectrogram(CImg<double> input);
CImg<double> gaussianFilter(double sigma,int filterSize);
CImg<double> meanFilter(int filterSize);
CImg<double> fft_gaus_filter(CImg<double> input, double sigma);

//Part 4 - Filtering
CImg<double> mark_image(const CImg<double> &input, int N);
CImg<double> check_image(const CImg<double> &input, int N);
void quantitativeTest(const CImg<double> &input,int N);
//Part 5
CImg<double> deconvolve(CImg<double> &input,CImg<double> &noisy);

// This code requires that input be a *square* image, and that each dimension
//  is a power of 2; i.e. that input.width() == input.height() == 2^k, where k
//  is an integer. You'll need to pad out your image (with 0's) first if it's
//  not a square image to begin with. (Padding with 0's has no effect on the FT!)
//
// Forward FFT transform: take input image, and return real and imaginary parts.
//
void fft(const CImg<double> &input, CImg<double> &fft_real, CImg<double> &fft_imag)
{
  fft_real = input;
  fft_imag = input;
  fft_imag = 0.0;

  FFT_2D(1, fft_real, fft_imag);
}

// Inverse FFT transform: take real and imaginary parts of fourier transform, and return
//  real-valued image.
//
void ifft(const CImg<double> &input_real, const CImg<double> &input_imag, CImg<double> &output_real)
{
  output_real = input_real;
  CImg<double> output_imag = input_imag;

  FFT_2D(0, output_real, output_imag);
}
CImg<double> averageGrayscale(CImg<double> input){
	//Creates a new grayscale image with same size as the input initialized to all 0s (black)
	CImg<double> output(input.width(), input.height(), 1, 1); 
	int i=0,j=0;
	//new grayscale image
	for(i=0;i<input.width();i++)
	{
		for(j=0;j<input.height();j++)
		{
			double avg = (0.3*input(i,j,0,0)) + (0.6*input(i,j,0,1)) + (0.1*input(i,j,0,2));
			output(i,j) = avg;
		}
	}
	return output;
}

int main(int argc, char **argv)
{
  try {

    if(argc < 4)
      {
			cout << "Insufficent number of arguments; correct usage:" << endl;
			cout << "p2 problemID inputfile outputfile" << endl;
			return -1;
      }
    
    char* part = argv[1];
    char* inputFile = argv[2];
	char* outputFile = argv[3];
	char* inputFile1;
	char* noisyImage;
	if(!strcmp(part, "2")||!strcmp(part, "5"))
    {
		inputFile1 = argv[3];
		outputFile = argv[4];
		cout << "In: " << inputFile << "In: " <<inputFile1<<"  Out: " << outputFile << endl;
	}
	else
	{
		outputFile = argv[3];
		cout << "In: " << inputFile <<"  Out: " << outputFile << endl;
	}
    
    CImg<double> input_image(inputFile);
	
	CImg<double> output;
    
    if(!strcmp(part,"2"))
      {
		cout << "# Problem 2 - Morphing" << endl;
		CImg<double> input_image1(inputFile1);
		if(argc != 6){	cout << "INPUT ERROR: Provide sigma as additional argument!" << endl;return -1;}
		if(input_image.spectrum() != 1)
			input_image = averageGrayscale(input_image);
		if(input_image1.spectrum() != 1)
			input_image1 = averageGrayscale(input_image1);
		output = morph(input_image,input_image1,atof(argv[5]));
		output.save(outputFile);
      }
	else if(!strcmp(part,"3.1"))
      {
		cout << "# Problem 3.1 - Spectrogram" << endl;
		if(argc != 4){cout << "INPUT ERROR: Provide output file name as additional argument!" << endl;return -1;}
		if(input_image.spectrum() != 1)
			input_image = averageGrayscale(input_image);
		output = spectrogram(input_image);
		output.save(outputFile);
	  }
	 else if(!strcmp(part,"3.2"))
	 {
		cout << "# Problem 3.2 - Remove interference" << endl;
		//if(input_image.spectrum() != 3){cout << "INPUT ERROR: Input image is not a color image!" << endl;return -1;}
		if(argc != 4){cout << "INPUT ERROR: Provide output file name as additional argument!" << endl;return -1;}
		if(input_image.spectrum() != 1)
			input_image = averageGrayscale(input_image);
		output = remove_interference(input_image);
		output.save(outputFile);
	 }
	 else if(!strcmp(part,"3.3"))
	 {
		cout << "# Problem 3.3 - Gaussian Filtering" << endl;
		//if(input_image.spectrum() != 3){cout << "INPUT ERROR: Input image is not a color image!" << endl;return -1;}
		if(argc != 5){cout << "INPUT ERROR: Provide sigma as additional argument!" << endl;return -1;}
		if(input_image.spectrum() != 1)
			input_image = averageGrayscale(input_image);
		output =  fft_gaus_filter(input_image,atof(argv[4]));
		output.save(outputFile);
	 }
	 else if(!strcmp(part,"3.4a"))
	 {
		cout << "# Problem 3.4a - Defilter" << endl;
		if(argc != 5){cout << "INPUT ERROR: Provide filter size as additional argument!" << endl;return -1;}
		if(input_image.spectrum() != 1)
			input_image = averageGrayscale(input_image);
		output = fft_defilter(input_image,atof(argv[4]));
		output.save(outputFile);
	 }
	 else if(!strcmp(part,"3.4b"))
	 {
		cout << "# Problem 3.4b - Defilter" << endl;
		if(argc != 5){cout << "INPUT ERROR: Provide sigma as additional argument!" << endl;return -1;}
		if(input_image.spectrum() != 1)
			input_image = averageGrayscale(input_image);
		output = fft_defilter_mean(input_image,atof(argv[4]));
		output.save(outputFile);
	 }
	 else if(!strcmp(part,"4.1"))
	 {
		cout << "# Problem 4.1 - Watermarking" << endl;
		if(argc != 5){cout << "INPUT ERROR: Provide key as additional argument!" << endl;return -1;}
		if(input_image.spectrum() != 1)
			input_image = averageGrayscale(input_image);
		output = mark_image(input_image,atof(argv[4]));
		output.save(outputFile);
	 }
	 else if(!strcmp(part,"4.2"))
	 {
		cout << "# Problem 4.2 - Testing" << endl;
		if(argc != 5){cout << "INPUT ERROR: Provide key as additional argument!" << endl;return -1;}
		if(input_image.spectrum() != 1)
			input_image = averageGrayscale(input_image);
		output = check_image(input_image,atof(argv[4]));
		output.save(outputFile);
	 }
	 else if(!strcmp(part,"5"))
	 {
		cout << "# Problem 5 - Deconvolution" << endl;
		noisyImage = argv[3];
		CImg<double> noise_image(noisyImage);
		if(argc != 5){cout << "INPUT ERROR: Provide output file name as additional argument!" << endl;return -1;}
		output = deconvolve(input_image,noise_image);
		output.save(outputFile);
	 }
	 else{
		cout<<"\n select correct input option";
	 }

  } 
  catch(const string &err) {
    cerr << "Error: " << err << endl;
  }
  return 0;
}

//Part 2
int ImgBorderProcess(int ij, int wh)
{
	if (ij >= wh)
		return 2 * wh - 1 - ij;
	else
		return abs(ij);	//abs(): reflect across axis
}
CImg<double> filter(CImg<double> input, CImg<double> filter){

	//Creates a new image with same size as the input initialized to all 0s (black)
	CImg<double> output(input, "xyzc", 0); 

	//FIXME Convolve with filter
	int i, j, fi, fj, fkx, fky;
	double pixelSum;

	fkx = (filter.width() - 1) / 2;
	fky = (filter.height() - 1) / 2;

	for (j = 0; j < input.height(); j++)	//input img
		for (i = 0; i < input.width(); i++)
		{
			pixelSum = 0;
			for (fj = -fky; fj <= fky; fj++)	//filter img
				for (fi = -fkx; fi <= fkx; fi++)
				{
					pixelSum = pixelSum	+ (filter(fi + fkx, fj + fky) * input(ImgBorderProcess(i - fi, input.width()), ImgBorderProcess(j - fj, input.height())));	//abs(): reflect across axis
				}
			output(i, j) = pixelSum;
		}

	return output;
}
CImg<double> gaussianFilter(CImg<double> input, double sigma){

	//determine filtersize
	int filterSize = int(35/sigma);

	//Creates a new grayscale image (just a matrix) to be our filter 
	CImg<double> H(filterSize, filterSize, 1, 1); 

	//Fill filter values
	int i, j, kx, ky;
	double sigma_1 =  1.0 / (2.0 * sigma * sigma);

	kx = (H.width() - 1) / 2;
	ky = (H.height() - 1) / 2;

	for (j = -ky; j <= ky; j++)
		for (i = -kx; i <= kx; i++)
		{
			H(i + kx, j + ky) = sigma_1 * M_1_PI * (exp(-(i * i + j * j) * sigma_1));	//M_1_PI: 1 / PI
		}

	//Convolve with filter and return
	return filter(input,H);
}
CImg<double> morph(CImg<double> input,CImg<double> input1, double sigma)
{
	//Creates a new image with same size as the input initialized to all 0s (black)
	CImg<double> output(input.width(), input.height());
	CImg<double> filter1(input.width(), input.height());
	CImg<double> filter2(input.width(), input.height());
	int i=0,j=0;

	filter1 = gaussianFilter(input,sigma+1);
	filter2 = gaussianFilter(input1,sigma);
	for(i=0;i<input.width();i++)
	{
		for(j=0;j<input.height();j++)
		{
			filter2(j,i) = input1(j,i) - filter2(j,i);
			output(j,i) = filter2(j,i) + filter1(j,i);
		}
	}
	return output;
}
//Part 3
// 3.1
CImg<double> resize(CImg<double> input,int num)
{
	int diff_ht = 0,diff_wd=0,num1 = 0;
	int i=0,j=0,k=0,l=0;
	int n = num;
	int cnt=0;
	
	if((num&(num-1))==0)
	{
		num1 = num;
		diff_wd = num1 - input.width();
		diff_ht = num1 - input.height();
	}
	else
	{
		while(n/2)
		{
			n=n/2;
			cnt++;
		}
		num1 = pow(2,(cnt+1));
		diff_wd = num1 - input.width();
		diff_ht = num1 - input.height();
	}
	CImg<double> output(num1,num1);
	for(i=diff_ht/2;i<(input.height()+(diff_ht/2));i++)
	{
		for(j=diff_wd/2;j<(input.width()+(diff_wd/2));j++)
		{
			output(j,i) = input(j-diff_wd/2,i-diff_ht/2);
		}
	}
	
	return output;
}
CImg<double> fft_magnitude(const CImg<double> &fft_real, const CImg<double> &fft_imag)
{
	int i=0,j=0;
	CImg<double> output(fft_real, "xyzc", 255);
	
	for(i=0;i<fft_real.width();i++)
	{
		for(j=0;j<fft_real.height();j++)
		{
			output(i,j) = log(sqrt((fft_real(i,j)*fft_real(i,j)) + (fft_imag(i,j)*fft_imag(i,j))));
		}
	}
	
	return output.normalize(0,255);
}
CImg<double> spectrogram(CImg<double> input)
{
	CImg<double> input1(input, "xyzc", 0);
	int size = input.height()&(input.height()-1);
	input1 = input;
	
	//resize the image
	if((input.height()!=input.width())||(size !=0))
	{
		size = input.height()>input.width()?input.height():input.width();
		input1 = resize(input,size);
	}
	CImg<double> output(input1, "xyzc", 0);
	CImg<double> fft_real(input1, "xyzc", 0);
	CImg<double> fft_imag(input1, "xyzc", 0);
	
	fft(input1,fft_real,fft_imag);
	output = fft_magnitude(fft_real,fft_imag);
	
	return output;
}
//3.2
CImg<double> remove_interference(const CImg<double> &input)
{
	CImg<double> output(input, "xyzc", 0);
	CImg<double> outputTemp;
	CImg<double> outputTemp1(input, "xyzc", 0);
	CImg<double> fft_real;
	CImg<double> fft_imag;
	int i=0,j=0;
	
	//get fft of image
	fft(input,fft_real,fft_imag);

	//get spectrogram of image
	outputTemp = fft_magnitude(fft_real,fft_imag);
	outputTemp.save((char*) "noise1_spec1.png");
	
	//mask the noise
	for(i=156;i<163;i++)
	{
		for(j=156;j<161;j++)
		{
			fft_real(i,j) = 0;
			fft_imag(i,j) = 0;
			fft_real(194+i,196+j) = 0;
			fft_imag(194+i,196+j) = 0;
		}
	}
	
	//get clean image
	ifft(fft_real,fft_imag,output);
	output = output.normalize(0,255);
	
	//get spectogram of masked image
	fft(output,fft_real,fft_imag);
	outputTemp = fft_magnitude(fft_real,fft_imag);
	outputTemp.save((char*) "noise1_spec2.png");
		
	return output;
}

//3.3
CImg<double> gaussianFilter(double sigma,int filterSize)
{
    double sigmasq = sigma * sigma;
    int af = ((filterSize + 1) / 2) - 1; //adjustementFactor
    double twoPiSigmaSq = 2.0 * M_PI * sigmasq;
 
    //Creates a new grayscale image (just a matrix) to be our filter 
    CImg<double> H(filterSize, filterSize, 1, 1);
    

    for (int i = 0; i < filterSize; i++)
    {
        for (int j = 0; j < filterSize; j++)
        {
            int newi = i - af;
            int newj = j - af;
            H(i, j) = (1.0 / twoPiSigmaSq) * exp((newi * newi + newj * newj) / (-2 * sigmasq));
		}
    }
	
    return H.normalize(0,255);
}
CImg<double> meanFilter(int filterSize){

	//Creates a new grayscale image (just a matrix) to be our filter
	CImg<double> H(filterSize, filterSize, 1, 1); 

	//FIXME Fill filter values
	int i, j;

	for (j = 0; j < H.height(); j++)
		for (i = 0; i < H.width(); i++)
		{
			H(i, j) = 1.0 / (filterSize * filterSize);
		}

	//Return filter
	return H;
}
CImg<double> fft_filter(const CImg<double> &input, const CImg<double> &filter)
{
	CImg<double> output(input, "xyzc", 0);
	CImg<double> output2(input, "xyzc", 0);
	CImg<double> outputTemp(input, "xyzc", 0);
	CImg<double> fft_real(input, "xyzc", 0),fft_real1(input, "xyzc", 0),fft_real2(input, "xyzc", 0);
	CImg<double> fft_imag(input, "xyzc", 0),fft_imag1(input, "xyzc", 0),fft_imag2(input, "xyzc", 0);
		
	int i=0,j=0;
	double tmp13,tmp24;
	int max_len = input.height();
	
	fft(input,fft_real,fft_imag);
	fft(filter,fft_real1,fft_imag1);
	
	for(i=0;i<max_len;i++)
	{
		for(j=0;j<max_len;j++)
		{
		    fft_real2(i,j) = (fft_real(i,j) * fft_real1(i,j)) - (fft_imag(i,j) * fft_imag1(i,j));
			fft_imag2(i,j) = (fft_real(i,j) * fft_imag1(i,j)) + (fft_real1(i,j) * fft_imag(i,j));
		}
	}
   	
	ifft(fft_real2,fft_imag2,output2);
	output2.normalize(0,255);
	output = output2;

	for (int i = 0; i<max_len/2; i++)
	{
		for (int k = 0; k<max_len/2; k++)
		{
          tmp13         = output(i,k);
          output(i,k)   = output(i+max_len/2,k+max_len/2);
          output(i+max_len/2,k+max_len/2) = tmp13;
		  
		  tmp24         = output(i,k+max_len/2);
          output(i,k+max_len/2)  = output(i+max_len/2,k);
          output(i+max_len/2,k)  = tmp24;
		}
	}
		
	return output;
}

CImg<double> fft_gaus_filter(CImg<double> input,double sigma)
{
	CImg<double> gaus_filter(input, "xyzc", 0);
	CImg<double> output(input, "xyzc", 0);
	CImg<double> input1(input, "xyzc", 0);
	CImg<double> outputTemp;
	CImg<double> fft_real,fft_imag;
	int size;
	
	size = input.height()>=input.width()?input.height():input.width();
	input1 = resize(input,size);
	gaus_filter = gaussianFilter(sigma,input1.height());
	
	output = fft_filter(input1,gaus_filter);
	
	//Get Blurred Image spectrogram
	fft(output,fft_real,fft_imag);
	outputTemp = fft_magnitude(fft_real,fft_imag);
	outputTemp.save((char*)"3.3_blurred_spec.png");
		
	return output;
}

//3.4a Defilter
CImg<double> fft_defilter(const CImg<double> &input, double sigma)
{
	CImg<double> output(input, "xyzc", 0);
	CImg<double> input1(input, "xyzc", 0);
	CImg<double> gaus_filter(input, "xyzc", 0);
	CImg<double> output2(input, "xyzc", 0);
	CImg<double> outputTemp;
	int i=0,j=0;
	double tmp13,tmp24;
	
	//Blur the image
	input1 = fft_gaus_filter(input,sigma);
	//Get Gaussian Filter
	gaus_filter = gaussianFilter(sigma,input1.height());
	
	CImg<double> fft_real(input1, "xyzc", 0),fft_real1(input1, "xyzc", 0),fft_real2(input1, "xyzc", 0);
	CImg<double> fft_imag(input1, "xyzc", 0),fft_imag1(input1, "xyzc", 0),fft_imag2(input1, "xyzc", 0);
	int max_len = input1.height();	
	
	fft(input1,fft_real,fft_imag);
	fft(gaus_filter,fft_real1,fft_imag1);
	
	for(i=0;i<max_len;i++)
	{
		for(j=0;j<max_len;j++)
		{
			fft_real2(i,j) = ((fft_real(i,j) * fft_real1(i,j))+(fft_imag(i,j) * fft_imag1(i,j)))/(pow(fft_real1(i,j),2)+pow(fft_imag1(i,j),2));		
			fft_imag2(i,j) = ((fft_real1(i,j) * fft_imag(i,j))-(fft_real(i,j) * fft_imag1(i,j)))/(pow(fft_real1(i,j),2)+pow(fft_imag1(i,j),2));		
		}
	}
   	
	ifft(fft_real2,fft_imag2,output2);
	output = output2.normalize(0,255);
	
	for (int i = 0; i<max_len/2; i++)
	{
		for (int k = 0; k<max_len/2; k++)
		{
          tmp13         = output(i,k);
          output(i,k)   = output(i+max_len/2,k+max_len/2);
          output(i+max_len/2,k+max_len/2) = tmp13;
		  
		  tmp24         = output(i,k+max_len/2);
          output(i,k+max_len/2)  = output(i+max_len/2,k);
          output(i+max_len/2,k)  = tmp24;
		}
	}
	
	//Get Blurred Image spectrogram
	fft(output,fft_real,fft_imag);
	outputTemp = fft_magnitude(fft_real,fft_imag);
	outputTemp.save((char*)"3.4a_deblurred_spec.png");
	
	return output;
}
//3.4b Defilter
CImg<double> fft_defilter_mean(CImg<double> &input,int filterSize)
{
	CImg<double> output(input, "xyzc", 0);
	CImg<double> output2(input, "xyzc", 0);
	CImg<double> input1(input, "xyzc", 0);
	CImg<double> filter(input, "xyzc", 0); 
	CImg<double> fft_real(input, "xyzc", 0),fft_real1(input, "xyzc", 0);
	CImg<double> fft_imag(input, "xyzc", 0),fft_imag1(input, "xyzc", 0);
	int size,i,k,j;
	double tmp13,tmp24;
	
	size = input.height()>=input.width()?input.height():input.width();
	input1 = resize(input,size);
	filter = meanFilter(filterSize);
	filter = resize(filter,size);
	
	CImg<double> fft_real2(input1, "xyzc", 0),fft_imag2(input1, "xyzc", 0);
	//blur the image
	input1 = fft_filter(input1,filter);
	input1.save((char*) "Mean_blurred_img.png");
	
	//deblur the image	
	int max_len = input1.height();
	fft(input1,fft_real,fft_imag);
	fft(filter,fft_real1,fft_imag1);
	
	for(i=0;i<max_len;i++)
	{
		for(j=0;j<max_len;j++)
		{
			fft_real2(i,j) = ((fft_real(i,j) * fft_real1(i,j))+(fft_imag(i,j) * fft_imag1(i,j)))/(pow(fft_real1(i,j),2)+pow(fft_imag1(i,j),2));		
			fft_imag2(i,j) = ((fft_real1(i,j) * fft_imag(i,j))-(fft_real(i,j) * fft_imag1(i,j)))/(pow(fft_real1(i,j),2)+pow(fft_imag1(i,j),2));		
		}
	}
   	
	ifft(fft_real2,fft_imag2,output2);
	output = output2.normalize(0,255);
	
	for (int i = 0; i<max_len/2; i++)
	{
		for (int k = 0; k<max_len/2; k++)
		{
          tmp13         = output(i,k);
          output(i,k)   = output(i+max_len/2,k+max_len/2);
          output(i+max_len/2,k+max_len/2) = tmp13;
		  
		  tmp24         = output(i,k+max_len/2);
          output(i,k+max_len/2)  = output(i+max_len/2,k);
          output(i+max_len/2,k)  = tmp24;
		}
	}

	return output;
}
//Part 4- Water marking
CImg<double> mark_image(const CImg<double> &input, int N)
{
	CImg<double> output(input, "xyzc", 0);
	CImg<double> input1(input, "xyzc", 0);
	CImg<double> spectrum(input, "xyzc", 0);
	int num = 0,size; 
	int x,y,width,height,x1,y1;
	CImg<double> fft_real;
	CImg<double> fft_imag;
    CImg<double> fft_real1;
	CImg<double> fft_imag1;
    
	srand(N);
    num = rand()%100;
	int v[num];
	input1 = input;
	for(int i=0;i<num;i++)
	{
		v[i] = rand()%2;
	}
	
	width = input1.width();
	height = input1.height();

	fft(input1,fft_real,fft_imag);

	for(int i=0;i<num;i++)
	{
		x1 = ((width/2)+1)+floor(120*cos(((3.14*i)/num)+3.14));
		y1 = ((height/2)+1)+floor(120*sin(((3.14*i)/num)+3.14));
	    x = ((width/2)+1)+floor(120*cos((3.14*i)/num));
		y = ((height/2)+1)+floor(120*sin((3.14*i)/num));
		fft_real(x1,y1) = fft_real(x1,y1)+(10*abs(fft_real(x1,y1))*v[i]);
		fft_real(x,y) = fft_real(x,y)+(10*abs(fft_real(x,y))*v[i]);
	
	}
    spectrum = fft_magnitude(fft_real,fft_imag);
	ifft(fft_real,fft_imag,output);
	spectrum.save((char*) "4.1a10Spectrum.png");
	return output.normalize(0,255);
}

CImg<double> check_image(const CImg<double> &input, int N)
{
	CImg<double> output(input, "xyzc", 0);
	CImg<double> input1(input, "xyzc", 0);
	int num = 0;
	float si=0,sj=0; 
	float numerator=0,denom=0;
	float corr = 0;
	int x,y,x3,y3,width,height;
	float xmean = 0,ymean = 0;
	CImg<double> fft_real;
	CImg<double> fft_imag;
    float pthreshold = 0.4;
	float nthreshold = -0.4;
	srand(N);
    num = rand()%100;
	int v[num];
	float c[num];
	for(int i=0;i<num;i++)
	v[i] = rand()%2;
	input1 = mark_image(input,N);
	width = input1.width();
	height = input1.height();
	
	fft(input1,fft_real,fft_imag);
	for(int i = 0;i<num;i++)
	{
		x = ((width/2)+1)+floor(50*cos((3.14*i)/num));
		y = ((height/2)+1)+floor(50*sin((3.14*i)/num));
		c[i] = fft_real(x,y);
	}

	for(int i=0;i<num;i++)
	{
		xmean = xmean + v[i];
		ymean = ymean + c[i];
	}
	xmean = xmean/num;
	ymean = ymean/num;
	for(int i =0;i<num;i++)
	{
		numerator = numerator+ ((v[i] - xmean) * (c[i] - ymean));
		si = si + ((v[i]-xmean)*(v[i]-xmean));
		sj = sj + ((c[i]-ymean)*(c[i]-ymean));
	}
	denom = sqrt(si*sj);
	if(denom!=0)
	corr = numerator/denom;
	else
	corr = 0;
	cout<<"\n correlation is ="<<corr;
	if(corr>=pthreshold || corr<=nthreshold)
	cout<<"\n water mark exists";
	quantitativeTest(input1,N);
	return fft_magnitude(fft_real,fft_imag);
}

//Quantitative Tests
void quantitativeTest(const CImg<double> &input,int N)
{
	CImg<double> output(input, "xyzc", 0);
	CImg<double> input1(input, "xyzc", 0);
	int num = 0;
	float si=0,sj=0; 
	float numerator=0,denom=0;
	float corr = 0;
	int count = 0;
	float corrArray[100];
	int x,y,x3,y3,width,height;
	float xmean = 0,ymean = 0;
	float pthreshold = 0.2;
	float nthreshold = -0.2;
	CImg<double> fft_real;
	CImg<double> fft_imag;
	for(int k=0;k<100;k++)
	{
		srand(N+k);
		num = rand()%100;
		int v[num];
		int c[num];
		for(int i=0;i<num;i++)
		v[i] = rand()%2;
		input1 = input;
		width = input1.width();
		height = input1.height();
		fft(input1,fft_real,fft_imag);
		for(int i = 0;i<num;i++)
		{
			x = ((width/2)+1)+floor(50*cos((3.14*i)/num));
			y = ((height/2)+1)+floor(50*sin((3.14*i)/num));
			c[i] = fft_real(x,y);
		}

		for(int i=0;i<num;i++)
		{
			xmean = xmean + v[i];
			ymean = ymean + c[i];
		}
		xmean = xmean/num;
		ymean = ymean/num;
		for(int i =0;i<num;i++)
		{
			numerator = numerator+ ((v[i] - xmean) * (c[i] - ymean));
			si = si + ((v[i]-xmean)*(v[i]-xmean));
			sj = sj + ((c[i]-ymean)*(c[i]-ymean));
		}
		denom = sqrt(si*sj);
		if(xmean!=0&&ymean!=0)
		corr = numerator/denom;
		else
		corr = 0;
		corrArray[k] = corr;
		xmean = 0;
		ymean = 0;
	}	
	cout<<"\n Correlation test results";
	for(int i=0;i<100;i++)
	{
		if(corrArray[i]>=pthreshold || corrArray[i]<=nthreshold)
		count++;
	}
	cout<<"\n Number of false positives = "<<count;
}
//Part 5 Deconvolution
CImg<double> deconvolve(CImg<double> &input,CImg<double> &noisy)
{
	CImg<double> output(input, "xyzc", 0);
	CImg<double> filter(input, "xyzc", 0);
	CImg<double> g_filter(input, "xyzc", 0);
	CImg<double> fft_real(input.height(),input.height());
	CImg<double> fft_real1(input.height(),input.height());
	CImg<double> fft_imag(input.height(),input.height());
	CImg<double> fft_imag1(input.height(),input.height());
	CImg<double> fft_real2(input.height(),input.height());
	CImg<double> fft_imag2(input.height(),input.height());
	CImg<double> fft_real3(input.height(),input.height());
	CImg<double> fft_imag3(input.height(),input.height());
	int i=0,j=0;
	int max_len = input.height();
	
	//adding gaussian noise
	/*g_filter = fft_gaus_filter(noisy,1.8);
	noisy = g_filter;*/
	
	fft(noisy,fft_real,fft_imag);
	fft(input,fft_real1,fft_imag1);
		
	//get kernel by dividing noisy image by input image
	for(i=0;i<max_len;i++)
	{
		for(j=0;j<max_len;j++)
		{
			fft_real2(i,j) = ((fft_real(i,j) * fft_real1(i,j))+(fft_imag(i,j) * fft_imag1(i,j)))/(pow(fft_real1(i,j),2)+pow(fft_imag1(i,j),2));		
			fft_imag2(i,j) = ((fft_real1(i,j) * fft_imag(i,j))-(fft_real(i,j) * fft_imag1(i,j)))/(pow(fft_real1(i,j),2)+pow(fft_imag1(i,j),2));			
		}
	}
	//check spectrogram of kernel
	filter = fft_magnitude(fft_real2,fft_imag2);
	filter.save((char*) "Motion_kernel.png");
	
	//get input by dividing noisy image by kernel
	for(i=0;i<max_len;i++)
	{
		for(j=0;j<max_len;j++)
		{
			fft_real3(i,j) = ((fft_real(i,j) * fft_real2(i,j))+(fft_imag(i,j) * fft_imag2(i,j)))/(pow(fft_real2(i,j),2)+pow(fft_imag2(i,j),2));		
			fft_imag3(i,j) = ((fft_real2(i,j) * fft_imag(i,j))-(fft_real(i,j) * fft_imag2(i,j)))/(pow(fft_real2(i,j),2)+pow(fft_imag2(i,j),2));				
		}
	}
	
	//get input clean image
	ifft(fft_real3,fft_imag3,output);
	
	return output.normalize(0,255);
}