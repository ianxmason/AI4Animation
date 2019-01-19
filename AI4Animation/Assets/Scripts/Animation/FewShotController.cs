using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

[System.Serializable]
public class FewShotController {

	public bool Inspect = false;

	public KeyCode MoveForward = KeyCode.W;
	public KeyCode MoveBackward = KeyCode.S;
	public KeyCode MoveLeft = KeyCode.A;
	public KeyCode MoveRight = KeyCode.D;
	public KeyCode TurnLeft = KeyCode.Q;
	public KeyCode TurnRight = KeyCode.E;
    public Gait[] Gaits = new Gait[0];
	public Style[] Styles = new Style[0];
	private string line; 
	private string[] style_list = new string[58];
	private string[] style_count = new string[58];
    private float[] style_walk = new float[58];
	private float[] style_run = new float[58];
	public FewShotController() {

	}
	
	public Vector3 QueryMove() {
		Vector3 move = Vector3.zero;
		if(Input.GetKey(MoveForward)) {
			move.z += 1f;
		}
		if(Input.GetKey(MoveBackward)) {
			move.z -= 1f;
		}
		if(Input.GetKey(MoveLeft)) {
			move.x -= 1f;
		}
		if(Input.GetKey(MoveRight)) {
			move.x += 1f;
		}
		return move;
	}

	
	public float QueryTurn() {
		float turn = 0f;
		if(Input.GetKey(TurnLeft)) {
			turn -= 1f;
		}
		if(Input.GetKey(TurnRight)) {
			turn += 1f;
		}
		return turn;
	}

	public void SetGaitCount(int count) {
		count = Mathf.Max(count, 0);
		if(Gaits.Length != count) {
			int size = Gaits.Length;
			System.Array.Resize(ref Gaits, count);
			for(int i=size; i<count; i++) {
				Gaits[i] = new Gait();
			}
		}
	}

    public void SetStyleCount(int count) {
		count = Mathf.Max(count, 0);
		if(Styles.Length != count) {
			int size = Styles.Length;
			System.Array.Resize(ref Styles, count);
			for(int i=size; i<count; i++) {
				Styles[i] = new Style();
			}
		}
	}

	public bool QueryAny() {
		for(int i=0; i<Gaits.Length; i++) {
			if(Gaits[i].Query()) {
				return true;
			}
		}
		return false;
	}

	[System.Serializable]
	public class Gait {
		public string Name;
		public float Bias = 1f;
		public KeyCode[] Keys = new KeyCode[0];
		public bool[] Negations = new bool[0];
		public Multiplier[] Multipliers = new Multiplier[0];

		public bool Query() {
			if(Keys.Length == 0) {
				return false;
			}

			bool active = false;

			for(int i=0; i<Keys.Length; i++) {
				if(!Negations[i]) {
					if(Keys[i] == KeyCode.None) {
						if(!Input.anyKey) {
							active = true;
						}
					} else {
						if(Input.GetKey(Keys[i])) {
							active = true;
						}
					}
				}
			}

			for(int i=0; i<Keys.Length; i++) {
				if(Negations[i]) {
					if(Keys[i] == KeyCode.None) {
						if(!Input.anyKey) {
							active = false;
						}
					} else {
						if(Input.GetKey(Keys[i])) {
							active = false;
						}
					}
				}
			}

			return active;
		}

		public bool QueryFwd() {

			bool active = false;

			for(int i=0; i<Keys.Length; i++) {
				if(Negations[i]) {
					if (Input.GetKey(Keys[i])) {
						active = false;
					}
					else {
						active = true;
					}
				}
			}

			return active;
		}

		public void SetKeyCount(int count) {
			count = Mathf.Max(count, 0);
			if(Keys.Length != count) {
				System.Array.Resize(ref Keys, count);
				System.Array.Resize(ref Negations, count);
			}
		}

		public void AddMultiplier() {
			ArrayExtensions.Add(ref Multipliers, new Multiplier());
		}

		public void RemoveMultiplier() {
			ArrayExtensions.Shrink(ref Multipliers);
		}

		[System.Serializable]
		public class Multiplier {
			public KeyCode Key;
			public float Value;
		}
	}


    [System.Serializable]
	public class Style {
		public string Name;
		public Gait[] Gaits = new Gait[3]; //Walk, Run, Stand
	}

	#if UNITY_EDITOR
	public void Inspector() {
		Utility.SetGUIColor(Color.grey);
		using(new GUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();
			if(Utility.GUIButton("Style Controller", UltiDraw.DarkGrey, UltiDraw.White)) {
				Inspect = !Inspect;
			}

			if(Inspect) {
				using(new EditorGUILayout.VerticalScope ("Box")) {
					MoveForward = (KeyCode)EditorGUILayout.EnumPopup("Move Forward", MoveForward);
					MoveBackward = (KeyCode)EditorGUILayout.EnumPopup("Move Backward", MoveBackward);
					MoveLeft = (KeyCode)EditorGUILayout.EnumPopup("Move Left", MoveLeft);
					MoveRight = (KeyCode)EditorGUILayout.EnumPopup("Move Right", MoveRight);
					TurnLeft = (KeyCode)EditorGUILayout.EnumPopup("Turn Left", TurnLeft);
					TurnRight = (KeyCode)EditorGUILayout.EnumPopup("Turn Right", TurnRight);
			
					// Read style information 
					int counter = 0;
					System.IO.StreamReader file =   
						new System.IO.StreamReader(@"Assets/Demo/style_cycle_counts.txt");  
					while((line = file.ReadLine()) != null)  
					{  
						string[] substrings = line.Split('\t');
						style_list[counter] = substrings[0];
						style_count[counter] = substrings[1]; // Removes the new line character
						counter++;
					}  
					file.Close(); 

                    counter = 0;
					System.IO.StreamReader file2 =   
						new System.IO.StreamReader(@"Assets/Demo/style_bias.txt"); 
					while((line = file2.ReadLine()) != null)  
					{  
						string[] substrings = line.Split('\t');
						style_walk[counter] = float.Parse(substrings[1]);
						style_run[counter] = float.Parse(substrings[2]); 
						counter++;
					}  
					file2.Close(); 

                    SetStyleCount(EditorGUILayout.IntField("Styles", style_list.Length)); 
					for(int i=0; i<Styles.Length; i++) {
						Utility.SetGUIColor(UltiDraw.Grey);
						using(new EditorGUILayout.VerticalScope ("Box")) {

							Utility.ResetGUIColor();
							Styles[i].Name = EditorGUILayout.TextField("Name", style_list[i]); 						

                            Styles[i].Gaits[0].Name = EditorGUILayout.TextField("Name", "Walk"); 
                            Styles[i].Gaits[0].Bias = EditorGUILayout.FloatField("Bias", (float)style_walk[i]);
						    Styles[i].Gaits[0].SetKeyCount(7);

                            Styles[i].Gaits[0].Keys[0] = (KeyCode)KeyCode.W;
                            Styles[i].Gaits[0].Negations[0] = false;
                            Styles[i].Gaits[0].Keys[1] = (KeyCode)KeyCode.A;
                            Styles[i].Gaits[0].Negations[1] = false;
                            Styles[i].Gaits[0].Keys[2] = (KeyCode)KeyCode.S;
                            Styles[i].Gaits[0].Negations[2] = false;
                            Styles[i].Gaits[0].Keys[3] = (KeyCode)KeyCode.D;
                            Styles[i].Gaits[0].Negations[3] = false;
                            Styles[i].Gaits[0].Keys[4] = (KeyCode)KeyCode.Q;
                            Styles[i].Gaits[0].Negations[4] = false;
                            Styles[i].Gaits[0].Keys[5] = (KeyCode)KeyCode.E;
                            Styles[i].Gaits[0].Negations[5] = false;
                            Styles[i].Gaits[0].Keys[6] = (KeyCode)KeyCode.LeftShift;
                            Styles[i].Gaits[0].Negations[6] = true;

                            Styles[i].Gaits[1].Name = EditorGUILayout.TextField("Name", "Run"); 
                            Styles[i].Gaits[1].Bias = EditorGUILayout.FloatField("Bias", (float)style_run[i]);
						    Styles[i].Gaits[1].SetKeyCount(1);

                            Styles[i].Gaits[1].Keys[0] = (KeyCode)KeyCode.LeftShift;
                            Styles[i].Gaits[1].Negations[0] = false;        

							Styles[i].Gaits[2].Name = EditorGUILayout.TextField("Name", "Stand"); 
                            Styles[i].Gaits[2].Bias = EditorGUILayout.FloatField("Bias", (float)0);
						    Styles[i].Gaits[2].SetKeyCount(1);

                            Styles[i].Gaits[2].Keys[0] = (KeyCode)KeyCode.W;
                            Styles[i].Gaits[2].Negations[0] = true;        
                    
                    
						}
					}
				}
			}
		}
	}
	#endif

}
