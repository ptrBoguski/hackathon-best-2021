using System;

namespace HackathonBEST
{
    public class EdgeDetector
    {
        public string FilePath { get; set; }

        public void Execute()
        {
            Console.WriteLine($"Executing edge detector for {FilePath}");
        }
    }
}