﻿using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepLearning {

    public enum TYPE {Vanilla, PFNN, PFNN_res_diag, PFNN_res_full, PFNN_res_cp, PFNN_CP};

    public abstract class Model : ScriptableObject {

		public string Folder = "Assets/Project/Parameters/";
        public Parameters Parameters;

        private List<Tensor> Tensors = new List<Tensor>();
        private List<string> Identifiers = new List<string>();

        public abstract void StoreParameters();
        public abstract void LoadParameters();
        public abstract void Predict();
        public abstract void SetInput(int index, float value);
        public abstract float GetOutput(int index);
		#if UNITY_EDITOR
		public abstract void Inspector();
		#endif

        public Tensor CreateTensor(int rows, int cols, string id) {
            if(Identifiers.Contains(id)) {
                Debug.Log("Tensor with ID " + id + " already contained.");
                return null;
            }
            Tensor T = new Tensor(rows, cols);
            Tensors.Add(T);
            Identifiers.Add(id);
            return T;
        }

        public Tensor CreateTensor(Parameters.FloatMatrix matrix, string id) {
            if(Identifiers.Contains(id)) {
                Debug.Log("Tensor with ID " + id + " already contained.");
                return null;
            }
            Tensor T = new Tensor(matrix.Rows, matrix.Cols);
            for(int x=0; x<matrix.Rows; x++) {
                for(int y=0; y<matrix.Cols; y++) {
                    T.SetValue(x, y, matrix.Values[x].Values[y]);
                }
            }
            Tensors.Add(T);
            Identifiers.Add(id);
            return T;
        }

        public void DeleteTensor(Tensor T) {
            int index = Tensors.IndexOf(T);
            if(index == -1) {
                Debug.Log("Tensor not found.");
                return;
            }
            Tensors.RemoveAt(index);
            Identifiers.RemoveAt(index);
            T.Delete();
        }

        public Tensor GetTensor(string id) {
            int index = Identifiers.IndexOf(id);
            if(index == -1) {
                return null;
            }
            return Tensors[index];
        }

        public string GetID(Tensor T) {
            int index = Tensors.IndexOf(T);
            if(index == -1) {
                return null;
            }
            return Identifiers[index];
        }

        public Tensor Normalise(Tensor IN, Tensor mean, Tensor std, Tensor OUT) {
            Eigen.Normalise(IN.Ptr, mean.Ptr, std.Ptr, OUT.Ptr);
            return OUT;
        }
        
        public Tensor Renormalise(Tensor IN, Tensor mean, Tensor std, Tensor OUT) {
            Eigen.Renormalise(IN.Ptr, mean.Ptr, std.Ptr, OUT.Ptr);
            return OUT;
        }

        public Tensor Layer(Tensor IN, Tensor W, Tensor b, Tensor OUT) {
            Eigen.Layer(IN.Ptr, W.Ptr, b.Ptr, OUT.Ptr);
            return OUT;
        }

        public Tensor Add(Tensor LHS, Tensor RHS, Tensor OUT) {
            Eigen.Add(LHS.Ptr, RHS.Ptr, OUT.Ptr);
            return OUT;
        }

        public Tensor Subtract(Tensor LHS, Tensor RHS, Tensor OUT) {
            Eigen.Subtract(LHS.Ptr, RHS.Ptr, OUT.Ptr);
            return OUT;
        }

        public Tensor Product(Tensor LHS, Tensor RHS, Tensor OUT) {
            Eigen.Product(LHS.Ptr, RHS.Ptr, OUT.Ptr);
            return OUT;
        }

        public Tensor PointwiseProduct(Tensor LHS, Tensor RHS, Tensor OUT) {
            Eigen.PointwiseProduct(LHS.Ptr, RHS.Ptr, OUT.Ptr);
            return OUT;
        }

        public Tensor Scale(Tensor LHS, float value, Tensor OUT) {
            Eigen.Scale(LHS.Ptr, value, OUT.Ptr);
            return OUT;
        }

        public float ColSum(Tensor T, int col) {
            return Eigen.ColSum(T.Ptr, col);
        }

        public Tensor Blend(Tensor T, Tensor W, float w) {
            Eigen.Blend(T.Ptr, W.Ptr, w);
            return T;
        }

        public Tensor ELU(Tensor T) {
            Eigen.ELU(T.Ptr);
            return T;
        }

        public Tensor Sigmoid(Tensor T) {
            Eigen.Sigmoid(T.Ptr);
            return T;
        }

        public Tensor TanH(Tensor T) {
            Eigen.TanH(T.Ptr);
            return T;
        }

        public Tensor SoftMax(Tensor T) {
            Eigen.SoftMax(T.Ptr);
            return T;
        }

    }
    
}