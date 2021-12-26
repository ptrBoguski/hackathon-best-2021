using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;
using Microsoft.Win32;
using Path = System.IO.Path;

namespace HackathonBEST
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        [DllImport("kernel32.dll")]
        private static extern bool AttachConsole(int dwProcessId);
        
        private DetectionMethod detectionMethod = DetectionMethod.CPU;
        private string currentFilePath;
        private double currentThreshold;

        public MainWindow()
        {
            InitializeComponent();
            AttachConsole(-1);
        }

        private void ImageDropZone_OnDrop(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                var filePaths = (string[]) e.Data.GetData(DataFormats.FileDrop);

                if (filePaths != null)
                {
                    LoadFile(filePaths[0]);
                }
            }
        }

        private void BrowseButton_OnClick(object sender, RoutedEventArgs e)
        {
            OpenFileDialog fileDialog = new OpenFileDialog();
            var result = fileDialog.ShowDialog();

            if (result ?? false)
            {
                LoadFile(fileDialog.FileName);
            }
        }

        private void LoadFile(string path)
        {
            DisplayInputImage(path);
            currentFilePath = path;
            var name = Path.GetFileName(path);
            FileNameLabel.Content = name;
        }
        

        private void DisplayInputImage(string path)
        {
            BitmapImage bitmap = new BitmapImage();  
            bitmap.BeginInit();  
            bitmap.UriSource = new Uri(path);  
            bitmap.EndInit();
            InputImageViewer.Source = bitmap;
        }

        private void ExecuteButton_OnClick(object sender, RoutedEventArgs e)
        {
            CurrentStatusText.Text = "Processing...";
            AllowUIToUpdate();
            var edgeDetector = detectionMethod.GetEdgeDetector();
            edgeDetector.OnDetectionCompleted += DetectionCompleted;
            edgeDetector.Execute(currentFilePath, currentThreshold);
        }
        
        void AllowUIToUpdate()
        {
            DispatcherFrame frame = new DispatcherFrame();
            Dispatcher.CurrentDispatcher.BeginInvoke(DispatcherPriority.Render, new DispatcherOperationCallback(delegate (object parameter)
            {
                frame.Continue = false;
                return null;
            }), null);

            Dispatcher.PushFrame(frame);
            Application.Current.Dispatcher.Invoke(DispatcherPriority.Background, new Action(delegate { }));
        }

        private void DetectionCompleted(BitmapImage image, TimeSpan duration)
        {
            CurrentStatusText.Text = "Ready";
            DisplayOutputImage(image);
            LastDurationText.Text = $"Last time: {duration.Milliseconds}ms";
        }
        
        private void DisplayOutputImage(BitmapImage image)
        {
            OutputImageViewer.Source = image;
        }

        private void ChangeDetectionMethod(DetectionMethod detectionMethod)
        {
            this.detectionMethod = detectionMethod;
        }

        private void UseCPUButton_OnClick(object sender, RoutedEventArgs e)
        {
            ChangeDetectionMethod(DetectionMethod.CPU);
        }

        private void UseGPUButton_OnClick(object sender, RoutedEventArgs e)
        {
            ChangeDetectionMethod(DetectionMethod.GPU);
        }

        private void SaveButton_OnClick(object sender, RoutedEventArgs e)
        {
            throw new NotImplementedException();
        }

        private void TresholdSlider_OnValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            currentThreshold = e.NewValue;
            Console.WriteLine(currentThreshold);
        }
    }
}