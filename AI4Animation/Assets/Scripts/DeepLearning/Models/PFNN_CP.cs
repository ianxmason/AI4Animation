using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepLearning {

	public class PFNN_CP : Model {

		public int XDim = 0;
		public int HDim = 0;
		public int YDim = 0;

		private Tensor Xmean, Xstd, Ymean, Ystd, W0_1, W1_1, W2_1, W0_3, W1_3, W2_3, b0, b1, b2, H1, H2, H3;
		private Tensor[] W0_2, W1_2, W2_2;
		private Tensor X, Y;

		private float Phase;

		private const float M_PI = 3.14159265358979323846f;

		public PFNN_CP() {
			
		}

		public override void StoreParameters() {
			Parameters = ScriptableObject.CreateInstance<Parameters>();
			Parameters.Store(Folder+"/Xmean.bin", XDim, 1);
			Parameters.Store(Folder+"/Xstd.bin", XDim, 1);
			Parameters.Store(Folder+"/Ymean.bin", YDim, 1);
			Parameters.Store(Folder+"/Ystd.bin", YDim, 1);

            Parameters.Store(Folder+"/W0_1.bin", 30, XDim);
            Parameters.Store(Folder+"/W1_1.bin", 30, HDim);
            Parameters.Store(Folder+"/W2_1.bin", 30, HDim);

            Parameters.Store(Folder+"/W0_3.bin", HDim, 30);
            Parameters.Store(Folder+"/W1_3.bin", HDim, 30);
            Parameters.Store(Folder+"/W2_3.bin", YDim, 30);

            Parameters.Store(Folder+"/b0.bin", HDim, 1);
            Parameters.Store(Folder+"/b1.bin", HDim, 1);
            Parameters.Store(Folder+"/b2.bin", YDim, 1);

			for(int i=0; i<50; i++) {
				Parameters.Store(Folder+"/W0_2_"+i.ToString("D3")+".bin", 30, 1);
				Parameters.Store(Folder+"/W1_2_"+i.ToString("D3")+".bin", 30, 1);
				Parameters.Store(Folder+"/W2_2_"+i.ToString("D3")+".bin", 30, 1);
			}
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

            W0_1 = CreateTensor(Parameters.Load(4), "W0_1");
            W1_1 = CreateTensor(Parameters.Load(5), "W1_1");
            W2_1 = CreateTensor(Parameters.Load(6), "W2_1");

            W0_3 = CreateTensor(Parameters.Load(7), "W0_3");
            W1_3 = CreateTensor(Parameters.Load(8), "W1_3");
            W2_3 = CreateTensor(Parameters.Load(9), "W2_3");

            b0 = CreateTensor(Parameters.Load(10), "b0");
            b1 = CreateTensor(Parameters.Load(11), "b1");
            b2 = CreateTensor(Parameters.Load(12), "b2");

			W0_2 = new Tensor[50];
			W1_2 = new Tensor[50];
			W2_2 = new Tensor[50];
			for(int i=0; i<50; i++) {
				W0_2[i] = CreateTensor(Parameters.Load(13 + i*3 + 0), "W0_2"+i);
				W1_2[i] = CreateTensor(Parameters.Load(13 + i*3 + 1), "W1_2"+i);
				W2_2[i] = CreateTensor(Parameters.Load(13 + i*3 + 2), "W2_2"+i);
			}

			X = CreateTensor(XDim, 1, "X");
			Y = CreateTensor(YDim, 1, "Y");
            H1 = CreateTensor(HDim, 1, "H1");
			H2 = CreateTensor(HDim, 1, "H2");
            H3 = CreateTensor(HDim, 1, "H3");

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
			Normalise(X, Xmean, Xstd, Y);

			//Process PFNN
			int index = (int)((Phase / (2f*M_PI)) * 50f);

            ELU(Add(Product(W0_3, PointwiseProduct(W0_2[index], Product(W0_1, Y, H1), H1), H1), b0, H1));
            ELU(Add(Product(W1_3, PointwiseProduct(W1_2[index], Product(W1_1, H1, H2), H2), H2), b1, H2));
            Add(Product(W2_3, PointwiseProduct(W2_2[index], Product(W2_1, H2, H3), H3), H3), b2, Y);

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