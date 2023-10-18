namespace NeuralNetwork;

public static class Program
{
    public static void Main()
    {
        // List<(Matrix, Matrix)> trainingData = new(){
        //     ( new([[0.,0.]]), new([0.])),
        //     // ( new([[0.,1.]]), new([1])),
        //     // ( new([[1.],[0]]), new([1])),
        //     // ( new([[1],[1]]), new([0]))
        // };
        var net = new Network(2, 2, 1);
        // net.SGD(trainingData, 50, 3);
        // net.PrintBiasesAndWeights();
    }
}