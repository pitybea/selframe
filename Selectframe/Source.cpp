
#include <opencv2/opencv.hpp>

#define CV_LIB_PATH "D:/downloads/opencv/build/x64/vc11/lib/"
#define CV_VER_NUM  CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define CV_EXT "d.lib"
#else
#define CV_EXT ".lib"
#endif

#pragma comment(lib, CV_LIB_PATH "opencv_calib3d"    CV_VER_NUM CV_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_contrib"    CV_VER_NUM CV_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_core"       CV_VER_NUM CV_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_features2d" CV_VER_NUM CV_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_flann"      CV_VER_NUM CV_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_gpu"        CV_VER_NUM CV_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_highgui"    CV_VER_NUM CV_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_imgproc"    CV_VER_NUM CV_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_legacy"     CV_VER_NUM CV_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_ml"         CV_VER_NUM CV_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_nonfree"    CV_VER_NUM CV_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_objdetect"  CV_VER_NUM CV_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_photo"      CV_VER_NUM CV_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_stitching"  CV_VER_NUM CV_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_ts"         CV_VER_NUM CV_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_video"      CV_VER_NUM CV_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_videostab"  CV_VER_NUM CV_EXT)


#include <stdlib.h>
#include <iostream>

#include <Windows.h>
#include <direct.h>
#include <functional>

#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <map>

using namespace std;

using namespace cv;


#include "../fileio/FileInOut.h"


	const int kptDet_maxCorners=200;
	const double kptDet_qualityLevel = 0.01;
	const double kptDet_minDistance = 8;
	const int kptDet_blockSize = 3;
	const bool kptDet_useHarrisDetector = false;
	const double kptDet_k = 0.04;
	const int kptTrack_iter=20;
	const double kptTrack_epsin=0.03;
	const int kptTrack_winsize=21;
	//const kptTrack_
	const int kptTrack_maxlevel=3;


	const double pnt_dis_threshold=5.0;

	const double static_pnt_cre=3.0;
	
template<class T>
static void FromSmall(vector<T>& p,int n,vector<int>& index)
{
	int k,j,i;
	T t;
	int ii;
	k=n/2;
	while(k>0)
	{
		for(j=k;j<=n-1;j++)
		{
			t=p[j];  ii=index[j];  i=j-k;
			while((i>=0)&&(p[i]>t))
			{
				p[i+k]=p[i];  index[i+k]=index[i];  i=i-k;
			}
			p[i+k]=t;  index[i+k]=ii;
		}
		k=k/2;
	}
};

void readRcdLst(char* inpf,vector<int>& rcd,map<string,int>& corres)
{
	FILE* fp=fopen(inpf,"r");
	//fopen_s(&fp,inpf,"r");
	if(fp!=NULL)
	{

		char tem[100];
		int rslt;
		while(fscanf(fp,"%s %d\n",&tem,&rslt)!=EOF)
		{
			string s(tem);
			if(corres.count(s))
			{
				rcd[corres[s]]=rslt;
			}
		
		}
	

		fclose(fp);
	}
}


int main_(int argc,char* argv[])
{
//	cout<<"whatever"<<endl;

//	getchar();
	#ifdef _DEBUG
	_chdir("D:\\DATA\\seiken0502\\sssf");
	#endif
	char* inp, *recd;

	inp="allimg.lst";

	recd="recd.lst";

	if(argc>1)
	{
		inp=argv[1];
		recd=argv[2];
	}

	vector<string> inpLst=fileIOclass::InVectorString(inp);

	map<string,int> nameIndxCorrespd;
	for (int i = 0; i < inpLst.size(); i++)
	{
		nameIndxCorrespd[inpLst[i]]=i;
	}

	vector<int> rcdLst;
	
	rcdLst.resize(inpLst.size(),-1);

	readRcdLst(recd,rcdLst,nameIndxCorrespd);
	vector<string> selLst;

	for (int i = 0; i < rcdLst.size(); i++)
	{
		if(rcdLst[i]==1)
			selLst.push_back(inpLst[i]);
	}


	FILE* fp=fopen("selFrm.lst","w");

	fprintf(fp,"%d\n",selLst.size());
	for(auto & s:selLst)
		fprintf(fp,"%s\n",s.c_str());
	fclose(fp);
	return 0;
}


int main(int argc,char* argv[])
{
//	cout<<"whatever"<<endl;

//	getchar();
	#ifdef _DEBUG
	_chdir("D:\\DATA\\seiken0502\\sssf");
	#endif
	char* inp, *recd;

	cout<<"selFrame ~A ~B (A is the list of images, default=allimg.lst) (B is the file for logging, default=new.lst)"<<endl;

	cout<<"press 'f' to change the flag\n press 's' to save\n learn other tricks by yourself"<<endl;

	inp="allimg.lst";

	recd="new.lst";

	if(argc>1)
	{
		inp=argv[1];
		recd=argv[2];
	}

	vector<string> inpLst=fileIOclass::InVectorString(inp);

	map<string,int> nameIndxCorrespd;
	for (int i = 0; i < inpLst.size(); i++)
	{
		nameIndxCorrespd[inpLst[i]]=i;
	}

	vector<int> rcdLst;
	
	rcdLst.resize(inpLst.size(),-1);

	readRcdLst(recd,rcdLst,nameIndxCorrespd);

	
	vector<int> bufferRcdLst=vector<int>(rcdLst);

	namedWindow("Window",1);
	namedWindow("flag",1);
	namedWindow("step",1);
	namedWindow("progress",1);
	namedWindow("monitor",1);

	Mat flagImgs[3];
	flagImgs[1]=Mat(Size(200,200),CV_8UC3,Scalar(87,1955,127));
	flagImgs[0]=Mat(Size(200,200),CV_8UC3,Scalar(155,129,243));
	flagImgs[2]=Mat(Size(200,200),CV_8UC3,Scalar(255,255,255));


	int flagOkOrNot=1;

	auto changeFlag=[&](int sig){
		if(sig!=-1)
			flagOkOrNot=(flagOkOrNot+1)%3;

		imshow("flag",flagImgs[(size_t)flagOkOrNot]);
	};


	

	int start_index=0;
	int step=1;

	int buffer_size=20;


	auto changeStep=[&](int direction)
	{
		if(direction>0)
		{
			step*=2;
		}
		else if(direction<0)
		{
			if(step>1)
				step/=2;
		}
		Mat pic = Mat::zeros(250,200,CV_8UC3);
		string show=to_string(step);
		int len=show.length();
		double sc=10.0/len;
		double sz=19*sc;
		int ht=(250-sz)/2;
		putText(pic, show,Point(0,250-ht), CV_FONT_HERSHEY_SIMPLEX, sc,  Scalar(255,255,255),2,8,false);
		imshow("step",pic);
	};

	for (int i  = 0; i  < rcdLst.size();  i ++)
	{
		if (rcdLst[i]==-1)
		{
			break;
		}
		++start_index;
	}


	FILE* fp=fopen(recd,"a");



	auto saveRcd=[&]()
	{
		assert(bufferRcdLst.size()==rcdLst.size());
		for (int i = 0; i < bufferRcdLst.size(); i++)
		{
			if (rcdLst[i]!=bufferRcdLst[i])
			{
				fprintf(fp,"%s %d\n", inpLst[i].c_str(),rcdLst[i]);
			}
		}
		bufferRcdLst=vector<int>(rcdLst);
	};

	int current_index=start_index;
	

	auto changeRcds=[&](int direction)
	{
		if(flagOkOrNot!=2)
		if(direction>0)
		{
			for (int i = current_index; i < current_index+step && i<rcdLst.size(); i++)
			{
				rcdLst[i]=flagOkOrNot;
			}

			printf("From %d to %d marked %d\n",current_index,current_index+step,flagOkOrNot);
		}
		else 
		{
			for (int i = current_index; i > current_index-step && i>=0; --i)
			{
				rcdLst[i]=-1;
			}
			printf("From %d to %d marked as unmarked(-1)\n",current_index,current_index-step);
		}
	};

	

	int recentLeftIndx=current_index;
	Mat mainWindowImg;
	Rect attentionArea=Rect(-1,-1,-1,-1);


	auto loadMainWinImg=[&]()
	{
		if((current_index<inpLst.size())&& (current_index>=0))
		{
			mainWindowImg=imread(inpLst[current_index]);
		
			imshow("Window",mainWindowImg);

		}
		if(attentionArea.width>0 && attentionArea.height>0)
		{
			rectangle(mainWindowImg,attentionArea,Scalar(0,0,255));
		}
		

		Mat pic = Mat::zeros(250,200,CV_8UC3);
		string show=to_string(current_index);
		int len=show.length();
		double sc=10.0/len;
		double sz=19*sc;
		int ht=(250-sz)/2;
		putText(pic, show,Point(0,250-ht), CV_FONT_HERSHEY_SIMPLEX, sc,  Scalar(255,255,255),2,8,false);
		imshow("progress",pic);

	};


	auto renderMonitor=[&]()
	{
		if((current_index<inpLst.size())&& (current_index>=0))
		if((recentLeftIndx<inpLst.size())&& (recentLeftIndx>=0))
		if (attentionArea.width>0 && attentionArea.height>0)
		{

			rectangle(mainWindowImg,attentionArea,Scalar(0,0,255));
			//imshow("Window",mainWindowImg);
			Mat img1,img2;
			img1=imread(inpLst[recentLeftIndx]);
			img2=imread(inpLst[current_index]);
			Mat fir=Mat(img1,attentionArea);
			Mat sec=Mat(img2,attentionArea);
			Mat src_gray;
			cvtColor( fir, src_gray, CV_BGR2GRAY );

			vector<Point2f> fea1(kptDet_maxCorners),fea2(kptDet_maxCorners);
		//	fea1.resize(kptDet_maxCorners);
			goodFeaturesToTrack( src_gray,
						fea1,
						kptDet_maxCorners,
						kptDet_qualityLevel,
						kptDet_minDistance,
						Mat(),
						kptDet_blockSize,
						kptDet_useHarrisDetector,
						kptDet_k );

			vector<uchar> status;
			vector<float> err;
			TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, kptTrack_iter, kptTrack_epsin);
			calcOpticalFlowPyrLK(fir,sec,
					fea1,fea2,status,err,
					Size(kptTrack_winsize,kptTrack_winsize),
					kptTrack_maxlevel, termcrit, 0, 0.001);

			vector<double> dis(fea1.size());
			vector<int> ind(fea1.size());
			for (int i = 0; i < dis.size(); i++)
			{
				dis[i]=norm(fea1[i]-fea2[i]);
				ind[i]=i;
			}
			FromSmall(dis,dis.size(),ind);
			int usefulSiz=dis.size()*0.6;
			double totalDis=0.0;
			for (int i  = 0; i  < usefulSiz; i ++)
			{
				totalDis+=dis[ind[i]];
			} 

			totalDis/=0.0001+usefulSiz;
			Mat pic = Mat::zeros(250,200,CV_8UC3);
			string show=to_string(totalDis);
			if(show.length()>6)
				show=show.substr(0,6);
			int len=show.length();
			double sc=10.0/len;
			double sz=19*sc;
			int ht=(250-sz)/2;
			putText(pic, show,Point(0,250-ht), CV_FONT_HERSHEY_SIMPLEX, sc,  Scalar(255,255,255),2,8,false);
			imshow("monitor",pic);

		}
	};
//	function<void()> pnt=renderMonitor;
	/*
	function<void(int,int,int,int, void*)> mouseFunc = [&](int e,int x,int y,int d, void* pt)
	{
		if(e==EVENT_LBUTTONDOWN)
		{

		}
		else if(e==EVENT_LBUTTONUP)
		{

		}
	};*/

	Point temAA;

	function<void(int,int,int)> passF=[&](int sig,int x,int y)
	{
		if (sig<0)
		{
			temAA.x=x;
			temAA.y=y;
		}
		else if (sig>0)
		{
			if (x>temAA.x)
			{
				attentionArea.x=temAA.x;
			}
			else
			{
				attentionArea.x=x;
			}
			attentionArea.width=abs(x-temAA.x);
			if(y>temAA.y)
			{
				attentionArea.y=temAA.y;
			}
			else
			{
				attentionArea.y=y;
			}
			attentionArea.height=abs(y-temAA.y);
			loadMainWinImg();
			renderMonitor();
		}
	};


	setMouseCallback("Window",[](int e,int x,int y,int d, void* pt)
	{
		auto p=*(function<void(int,int,int)>*)pt;
		//auto q=*p;
		if(e==EVENT_LBUTTONDOWN)
		{
			p(-1,x,y);
		}
		else if(e==EVENT_LBUTTONUP)
		{
			p(1,x,y);
		}
	},&passF);


	changeStep(0);
	changeFlag(-1);





	loadMainWinImg();
	while((current_index<inpLst.size())&& (current_index>=0))
	{


		
		
		auto keyPressed =	waitKey(0);

	//	cout<<keyPressed<<endl;

		switch ( keyPressed)
		{
			/*
			Upkey : 2490368
			DownKey : 2621440
			LeftKey : 2424832
			RightKey: 2555904
			*/
		case 2555904:
			recentLeftIndx=current_index;
			changeRcds(1);

			current_index+=step;	
			
			renderMonitor();
//			mainWindowImg=imread(inpLst[current_index]);
			loadMainWinImg();
			break;
		case 2424832:
			changeRcds(-1);
			current_index-=step;
			if(current_index<recentLeftIndx)
				if(current_index>0)
					recentLeftIndx=current_index-1;
				else
					recentLeftIndx=0;

			
			renderMonitor();
			//mainWindowImg=imread(inpLst[current_index]);
			loadMainWinImg();
			break;
		case 2490368:
			changeStep(1);
			break;
		case 2621440:
			changeStep(-1);
			break;

		case 'f':
			changeFlag(0);
			break;


		case 's':
			saveRcd();
			break;

		case 27:
			goto endflag;
			break;
		default:
			break;
		}
	
	}

endflag:

	fclose(fp);
//	getchar();

	return 0;
}