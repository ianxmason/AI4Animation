using UnityEngine;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepLearning {

	public class PFNN_res_cp : Model {

		public int XDim = 0;
		public int HDim = 0;
		public int YDim = 0;

		private Tensor Xmean, Xstd, Ymean, Ystd;
		private Tensor[] W0, W1, W2, b0, b1, b2, temp_tens, sty_W0, sty_W2, sty_b;
		private List<Tensor[]> sty_W1;
		private Tensor X, H1, H2, Y;

		private string line; 
		private string[] style_list = new string[58];
		private string[] style_count = new string[58];
		private string[] switch_list = new string[58];

		private float Phase;

		private const float M_PI = 3.14159265358979323846f;

		public PFNN_res_cp() {
			int counter = 0;
			System.IO.StreamReader file =   
				new System.IO.StreamReader(@"Assets/Demo/style_cycle_counts.txt"); 
			while((line = file.ReadLine()) != null)  
			{  
				string[] substrings = line.Split('\t');
				style_list[counter] = "/Fewshot/"+substrings[0];
				switch_list[counter] = substrings[0];
				style_count[counter] = substrings[1];
				counter++;
			}  
			file.Close(); 

			// Due to inconsistencies we need this manual correction
			style_list[0] = "/ang";
			style_list[1] = "/chi";
			style_list[2] = "/dep";
			style_list[3] = "/neu";
			// style_list[3] = "/ang";
			style_list[4] = "/old";
			style_list[5] = "/pro";
			style_list[6] = "/sex";
			style_list[7] = "/str";

			switch_list[0] = "Angry";
			switch_list[1] = "Childlike";
			switch_list[2] = "Depressed";
			switch_list[3] = "Neutral";
			switch_list[4] = "Old";
			switch_list[5] = "Proud";
			switch_list[6] = "Sexy";
			switch_list[7] = "Strutting";
		}

		public override void StoreParameters() {
			Parameters = ScriptableObject.CreateInstance<Parameters>();
			Parameters.Store(Folder+"/Xmean.bin", XDim, 1);
			Parameters.Store(Folder+"/Xstd.bin", XDim, 1);
			Parameters.Store(Folder+"/Ymean.bin", YDim, 1);
			Parameters.Store(Folder+"/Ystd.bin", YDim, 1);
			for(int i=0; i<50; i++) {
				Parameters.Store(Folder+"/W0_"+i.ToString("D3")+".bin", HDim, XDim);
				Parameters.Store(Folder+"/W1_"+i.ToString("D3")+".bin", HDim, HDim);
				Parameters.Store(Folder+"/W2_"+i.ToString("D3")+".bin", YDim, HDim);
				Parameters.Store(Folder+"/b0_"+i.ToString("D3")+".bin", HDim, 1);
				Parameters.Store(Folder+"/b1_"+i.ToString("D3")+".bin", HDim, 1);
				Parameters.Store(Folder+"/b2_"+i.ToString("D3")+".bin", YDim, 1);
			}


			for(int i=0; i<58; i++){
			// for(int i=0; i<17; i++){
				Parameters.Store(Folder+style_list[i]+"_W0"+".bin", 30, HDim);
			}	

			for(int i=0; i<58; i++){
			// for(int i=0; i<17; i++){
				for(int j=0; j<50; j++) {
					Parameters.Store(Folder+style_list[i]+"_W1_"+j.ToString("D3")+".bin", 30, 1);
				}
			}	

			for(int i=0; i<58; i++){
			// for(int i=0; i<17; i++){
				Parameters.Store(Folder+style_list[i]+"_W2"+".bin", HDim, 30);
			}

			for(int i=0; i<58; i++){
			// for(int i=0; i<17; i++){
				Parameters.Store(Folder+style_list[i]+"_b"+".bin", HDim, 1);
			}


			// for(int i=8; i<58; i++){
			// // for(int i=0; i<8; i++){
			// 	Parameters.Store(Folder+style_list[i]+"_W0"+".bin", 30, HDim);
			// }	

			// for(int i=8; i<58; i++){
			// // for(int i=0; i<8; i++){
			// 	for(int j=0; j<50; j++) {
			// 		Parameters.Store(Folder+style_list[i]+"_W1_"+j.ToString("D3")+".bin", 30, 1);
			// 	}
			// }	

			// for(int i=8; i<58; i++){
			// // for(int i=0; i<8; i++){
			// 	Parameters.Store(Folder+style_list[i]+"_W2"+".bin", HDim, 30);
			// }

			// for(int i=8; i<58; i++){
			// // for(int i=0; i<8; i++){
			// 	Parameters.Store(Folder+style_list[i]+"_b"+".bin", HDim, 1);
			// }
		}

		public override void LoadParameters() {
			if(Parameters == null) {
				Debug.Log("Building PFNN failed because no parameters were saved.");
				return;
			}

			Xmean = CreateTensor(Parameters.Load(0), "Xmean");
			Xstd = CreateTensor(Parameters.Load(1), "Xstd");
			Ymean = CreateTensor(Parameters.Load(2), "Ymean");
			Ystd = CreateTensor(Parameters.Load(3), "Ystd");

			W0 = new Tensor[50];
			W1 = new Tensor[50];
			W2 = new Tensor[50];
			b0 = new Tensor[50];
			b1 = new Tensor[50];
			b2 = new Tensor[50];

			sty_W0 = new Tensor[58];
			sty_W1 = new List<Tensor[]>(58);
			for(int i=0; i<58; i++){
				sty_W1.Add(new Tensor[50]);
			}
			sty_W2 = new Tensor[58];
			sty_b = new Tensor[58];

			// sty_W0 = new Tensor[8];
			// sty_W1 = new List<Tensor[]>(8);
			// for(int i=0; i<8; i++){
			// 	sty_W1.Add(new Tensor[50]);
			// }
			// sty_W2 = new Tensor[8];
			// sty_b = new Tensor[8];
			
			for(int i=0; i<50; i++) {
				W0[i] = CreateTensor(Parameters.Load(4 + i*6 + 0), "W0"+i);
				W1[i] = CreateTensor(Parameters.Load(4 + i*6 + 1), "W1"+i);
				W2[i] = CreateTensor(Parameters.Load(4 + i*6 + 2), "W2"+i);
				b0[i] = CreateTensor(Parameters.Load(4 + i*6 + 3), "b0"+i);
				b1[i] = CreateTensor(Parameters.Load(4 + i*6 + 4), "b1"+i);
				b2[i] = CreateTensor(Parameters.Load(4 + i*6 + 5), "b2"+i);
			}

			for(int i=0; i<58; i++){
			// for(int i=0; i<17; i++){
				sty_W0[i] = CreateTensor(Parameters.Load(4 + 6*50 + i), switch_list[i]+"_W0");
			}	

			for(int i=0; i<58; i++){
			// for(int i=0; i<17; i++){
				for(int j=0; j<50; j++) {
					sty_W1[i][j] = CreateTensor(Parameters.Load(4 + 6*50 + 58 + i*50 + j), switch_list[i]+"_W1_"+j);
					// sty_W1[i][j] = CreateTensor(Parameters.Load(4 + 6*50 + 17 + i*50 + j), switch_list[i]+"_W1_"+j);
				}
			}	

			for(int i=0; i<58; i++){
			// for(int i=0; i<17; i++){
				sty_W2[i] = CreateTensor(Parameters.Load(4 + 6*50 + 58 + 58*50 + i), switch_list[i]+"_W2");
				// sty_W2[i] = CreateTensor(Parameters.Load(4 + 6*50 + 17 + 17*50 + i), switch_list[i]+"_W2");
			}	

			for(int i=0; i<58; i++){
			// for(int i=0; i<17; i++){
				sty_b[i] = CreateTensor(Parameters.Load(4 + 6*50 + 58 + 58*50 + 58 + i), switch_list[i]+"_b");
				// sty_b[i] = CreateTensor(Parameters.Load(4 + 6*50 + 17 + 17*50 + 17 + i), switch_list[i]+"_b");
			}	

			// for(int i=8; i<58; i++){
			// // for(int i=0; i<8; i++){
			// 	sty_W0[i] = CreateTensor(Parameters.Load(4 + 6*50 + (i-8)), switch_list[i]+"_W0");
			// }	

			// for(int i=8; i<58; i++){
			// // for(int i=0; i<8; i++){
			// 	for(int j=0; j<50; j++) {
			// 		sty_W1[i][j] = CreateTensor(Parameters.Load(4 + 6*50 + 50 + (i-8)*50 + j), switch_list[i]+"_W1_"+j);
			// 		// sty_W1[i][j] = CreateTensor(Parameters.Load(4 + 6*50 + 8 + i*50 + j), switch_list[i]+"_W1_"+j);
			// 	}
			// }	

			// for(int i=8; i<58; i++){
			// // for(int i=0; i<8; i++){
			// 	sty_W2[i] = CreateTensor(Parameters.Load(4 + 6*50 + 50 + 50*50 + (i-8)), switch_list[i]+"_W2");
			// 	// sty_W2[i] = CreateTensor(Parameters.Load(4 + 6*50 + 8 + 8*50 + i), switch_list[i]+"_W2");
			// }	

			// for(int i=8; i<58; i++){
			// // for(int i=0; i<8; i++){
			// 	sty_b[i] = CreateTensor(Parameters.Load(4 + 6*50 + 50 + 50*50 + 50 + (i-8)), switch_list[i]+"_b");
			// 	// sty_b[i] = CreateTensor(Parameters.Load(4 + 6*50 + 8 + 8*50 + 8 + i), switch_list[i]+"_b");
			// }	
				
			X = CreateTensor(XDim, 1, "X");
			H1 = CreateTensor(HDim, 1, "H1");
			H2 = CreateTensor(HDim, 1, "H2");
			Y = CreateTensor(YDim, 1, "Y");

			Phase = 0f;
		}

		public override void SetInput(int index, float value) {
			X.SetValue(index, 0, value);
		}

		public override float GetOutput(int index) {
			return Y.GetValue(index, 0);
		}
		
		public override void Predict() {
			//Normalise Input
			Debug.Log("Using Predict function! Please use Predict_res instead");
		}

		public void Predict_res(string style) {
			//Normalise Input
			Normalise(X, Xmean, Xstd, Y);

			//Process PFNN
			int index = (int)((Phase / (2f*M_PI)) * 50f);
			
			ELU(Layer(Y, W0[index], b0[index], Y));

			int counter2 = 0;
			for(int i=0; i<switch_list.Length; i++){
				// if (style == switch_list[i]){
				// 	if (i==3){
				// 		ELU(Layer(Y, W1[index], b1[index], Y)); // 0 in res ad
				// 	}
				// 	else{
				// 		ELU(Add(Layer(Y, W1[index], b1[index], H1), Add(Product(sty_W2[i], PointwiseProduct(sty_W1[i][index], Product(sty_W0[i], Y, H2), H2), H2), sty_b[i], H2), Y));
				// 		// ELU(Layer(Y, W1[index], b1[index], Y)); // 0 in res ad
				// 		// ELU(Add(Product(sty_W2[i], PointwiseProduct(sty_W1[i][index], Product(sty_W0[i], Y, H2), H2), H2), sty_b[i], Y)); // Only resad turned on
				// 	}
				// }
				if (style == switch_list[i]){
					ELU(Add(Layer(Y, W1[index], b1[index], H1), Add(Product(sty_W2[i], PointwiseProduct(sty_W1[i][index], Product(sty_W0[i], Y, H2), H2), H2), sty_b[i], H2), Y));
					// ELU(Layer(Y, W1[index], b1[index], Y)); // 0 in res ad
					// ELU(Add(Product(sty_W2[i], PointwiseProduct(sty_W1[i][index], Product(sty_W0[i], Y, H2), H2), H2), sty_b[i], Y)); // Only resad turned on
				}
				else {
					counter2++;
				}
			}
			if (counter2 ==switch_list.Length){
				Debug.Log("No valid style selected!");
			}
			Layer(Y, W2[index], b2[index], Y);

			//Renormalise Output
			Renormalise(Y, Ymean, Ystd, Y);
		}

		/*
		private Matrix Linear(ref Matrix y0, ref Matrix y1, float mu) {
			return (1.0f-mu) * y0 + (mu) * y1;
		}

		private Matrix Cubic(ref Matrix y0, ref Matrix y1, ref Matrix y2, ref Matrix y3, float mu) {
			return
			(-0.5f*y0 + 1.5f*y1 - 1.5f*y2 + 0.5f*y3)*mu*mu*mu + 
			(y0 - 2.5f*y1 + 2.0f*y2 - 0.5f*y3)*mu*mu + 
			(-0.5f*y0 + 0.5f*y2)*mu + 
			(y1);
		}
		*/

		public void SetPhase(float value) {
			Phase = value;
		}

		public float GetPhase() {
			return Phase;
		}

		#if UNITY_EDITOR
		public override void Inspector() {
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Folder = EditorGUILayout.TextField("Folder", Folder);
				EditorGUILayout.BeginHorizontal();
				if(GUILayout.Button("Store Parameters")) {
					StoreParameters();
				}
				Parameters = (Parameters)EditorGUILayout.ObjectField(Parameters, typeof(Parameters), true);
				EditorGUILayout.EndHorizontal();

				XDim = EditorGUILayout.IntField("XDim", XDim);
				HDim = EditorGUILayout.IntField("HDim", HDim);
				YDim = EditorGUILayout.IntField("YDim", YDim);

				EditorGUILayout.Slider("Phase", Phase, 0f, 2f*Mathf.PI);
			}
		}
		#endif

	}

}
