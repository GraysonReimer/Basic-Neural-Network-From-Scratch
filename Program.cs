using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_01
{
    class Program
    {
        //List of Layers that make up the Neural Network
        public static Layer[] Layers = new Layer[] { new Layer(3), new Layer(4), new Layer(4), new Layer(2) };
        //Easy Access to the first layer of the network
        public static Layer Inputs => Layers[0];

        //Easy Access to the last layer of the network
        public static Layer Outputs => Layers[Layers.Length - 1];

        //The data that will be fed into the Network
        public static Data data;

        public static System.Random random = new System.Random();

        //Appends each layer to the previous layer and then randomizes the layers
        static void InitializeLayers()
        {
            if (Layers.Length >= 3)
            {
                for (int i = 1; i < Layers.Length; i++)
                {
                    Layers[i].AppendToLayer(Layers[i - 1]);
                }
                for (int i = 0; i < Layers.Length; i++)
                {
                    Layers[i].Randomize();
                }
            }
        }
        //Sets the data variable to an array of numbers 0 or 1. If last number is 1 then answer is 1.
        static void GenerateData()
        {
            data = new Data();
            data.Values = new float[Inputs.Length];
            for (int i = 0; i < data.Length; i++)
            {
                int NewData = random.Next(0, 2);
                data[i] = NewData;
                if (data[data.Length - 1] == 1)
                {
                    data.Answer = 1;
                }
                else
                {
                    data.Answer = 0;
                }
            }    
        }
        //Returns the Sigmoid of any inputed float
        public static float Sigmoid(float value)
        {
            float k = (float)Math.Exp(value);
            return k / (1.0f + k);
        }

        //Feed Data into the Network to get an output
        static void FeedForward()
        {
            GenerateData();
            Inputs.SetValues(data.Values);
            for (int i = 1; i < Layers.Length; i++)
            {
                Layer layer = Layers[i];
                int index = 0;
                foreach (Neuron neuron in layer.Neurons)
                {
                    float newValue = 0;
                    foreach (Neuron prevNeuron in Layers[i-1].Neurons)
                    {
                        newValue += prevNeuron.Value * prevNeuron.Weights[index];
                    }
                    newValue += neuron.Bias;

                    neuron.Value = Sigmoid(newValue);

                    index++;
                }
            }
        }
        //Adjust the biases and weights
        static void BackPropagate()
        {
            float LearningRate = 0.02f;
            float BiasLearningRate = 0.006f;
            float[] expectedAnswer;
            if (data.Answer == 1)
                expectedAnswer = new float[] { 0, 1 };
            else
                expectedAnswer = new float[] { 1, 0 };


            //Begin with changing the Weights connected to the output layer

            //foreach Neuron in Outputs
            for (int NeuronNum = 0; NeuronNum < Outputs.Length; NeuronNum++)
            {
                Neuron neuron = Outputs[NeuronNum];
                float Gamma = (neuron.Value - expectedAnswer[NeuronNum]) * (1 - (neuron.Value) * (neuron.Value));
                neuron.Gamma = Gamma;
                //foreach Neuron in Previous Layer
                for (int PrevLayerNum = 0; PrevLayerNum < Layers[Layers.Length - 2].Length; PrevLayerNum++)
                {
                    //Calculate a New Value to Subtract
                    float Delta = Gamma*Layers[Layers.Length - 2][PrevLayerNum].Value;
                    //Apply the New Value to Adjust the Weight thats pointing to the neuron
                    Layers[Layers.Length - 2][PrevLayerNum].Weights[NeuronNum] -= Delta * LearningRate;
                    //Apply the New Value to Adjust the Bias of the Previous Weight
                    Layers[Layers.Length - 2][PrevLayerNum].Bias -= Delta * BiasLearningRate;
                }
            }
            
            for (int LayerNum = Layers.Length - 2; LayerNum > 0; LayerNum--)
            {
                Layer layer = Layers[LayerNum];

                for (int NeuronNum = 0; NeuronNum < layer.Length; NeuronNum++)
                {
                    Neuron neuron = layer[NeuronNum];
                    float Gamma = 0;
                    for (int NeuronNum2 = 0; NeuronNum2 < Layers[LayerNum + 1].Length; NeuronNum2++)
                    {
                        Gamma += Layers[LayerNum + 1][NeuronNum2].Gamma * Layers[LayerNum][NeuronNum].Weights[NeuronNum2];
                    }
                    Gamma *= (1 - neuron.Value * neuron.Value);
                    neuron.Gamma = Gamma;
                    //foreach Neuron in Previous Layer
                    for (int PrevLayerNum = 0; PrevLayerNum < Layers[LayerNum-1].Length; PrevLayerNum++)
                    {
                        //Calculate a New Value to Subtract
                        float Delta = Gamma * Layers[LayerNum - 1][PrevLayerNum].Value;
                        //Apply the New Value to Adjust the Weight thats pointing to the neuron
                        Layers[LayerNum - 1][PrevLayerNum].Weights[NeuronNum] -= Delta * LearningRate;
                        //Apply the New Value to Adjust the Bias of the Previous Weight
                        Layers[LayerNum - 1][PrevLayerNum].Bias -= Delta * BiasLearningRate;
                    }
                }
            }

        }

        //Neuron 0 is for answer of 0. Neuron 1 is for answer of 1.

        //Feeds Forward and then back propagates for the specified number of iterations
        static void RunNetwork(int Iterations, bool log)
        {
            print("\n\nRunning Network...\n\n");
            for (int i = 0; i < Iterations; i++)
            {
                print($"Iteration {i+1} / {Iterations} Completed\n");
                FeedForward();
                BackPropagate();
            }

            void print(string str)
            {
                if (log)
                {
                    Console.WriteLine(str);
                }
            }
        }
        static void TestNetwork(int Iterations)
        {
            int CorrectAnswers = 0;
            for (int i = 0; i < Iterations; i++)
            {
                FeedForward();
                Console.WriteLine("\n\n0: " + Outputs[0].Value);
                Console.WriteLine("1: " + Outputs[1].Value);
                if ((data.Answer == 0 && Outputs[0].Value > Outputs[1].Value) || (data.Answer == 1 && Outputs[1].Value > Outputs[0].Value))
                {
                    Console.WriteLine("\nCorrect");
                    CorrectAnswers++;
                }
                else
                {
                    Console.WriteLine("\nIncorrect");
                }
            }
            Console.WriteLine($"\n\nNetwork Accuracy: {(float)CorrectAnswers/(float)Iterations*100f}%");

        }
        static void Main(string[] args)
        {
            InitializeLayers();
            RunNetwork(10000, true);
            TestNetwork(100);
            Console.ReadLine();
        }
    }
}
