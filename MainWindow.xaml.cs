using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
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
        private double currentSigma;
        private bool gaussianEnabled = false;
        private bool maxSupressionEnabled = false;
        private int[] currentXMask = {1,1,1,0,0,0,-1,-1,-1};
        private int[] currentYMask = {1,0,-1,1,0,-1,1,0,-1};
        private bool autoExecuteEnabled = false;
        private BitmapImage outputImage;

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
            Execute();
        }

        private void Execute()
        {
            CurrentStatusText.Text = "Processing...";
            AllowUIToUpdate();
            var edgeDetector = detectionMethod.GetEdgeDetector();
            edgeDetector.OnDetectionCompleted += DetectionCompleted;
            edgeDetector.Execute(currentFilePath, currentXMask, currentYMask, currentThreshold, currentSigma, gaussianEnabled, maxSupressionEnabled);
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
            outputImage = image;
            OutputImageViewer.Source = image;
        }

        private void ChangeDetectionMethod(DetectionMethod detectionMethod)
        {
            this.detectionMethod = detectionMethod;
        }

        private void UseCPUButton_OnClick(object sender, RoutedEventArgs e)
        {
            ChangeDetectionMethod(DetectionMethod.CPU);
            if (autoExecuteEnabled)
            {
                Execute();
            }
        }

        private void UseGPUButton_OnClick(object sender, RoutedEventArgs e)
        {
            ChangeDetectionMethod(DetectionMethod.GPU);
            if (autoExecuteEnabled)
            {
                Execute();
            }
        }

        private void SaveButton_OnClick(object sender, RoutedEventArgs e)
        {
            SaveFileDialog saveFileDialog = new SaveFileDialog();
            if (saveFileDialog.ShowDialog() == true)
            {
                BitmapEncoder encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(outputImage));

                using (var fileStream = new System.IO.FileStream(saveFileDialog.FileName + ".png", System.IO.FileMode.Create))
                {
                    encoder.Save(fileStream);
                }
            }
        }

        private void TresholdSlider_OnValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            currentThreshold = e.NewValue;
            if (autoExecuteEnabled)
            {
                Execute();
            }
        }

        private void ModifyMaskButton_OnClick(object sender, RoutedEventArgs e)
        {
            MaskEditor maskEditor = new MaskEditor(
                currentXMask,
                currentYMask,
                newMask=>
                {
                    currentXMask = newMask;
                    if (autoExecuteEnabled)
                    {
                        Execute();
                    }
                },
                newMask =>
                {
                    currentYMask = newMask;
                    if (autoExecuteEnabled)
                    {
                        Execute();
                    }
                });
            maskEditor.Owner = this;
            maskEditor.ShowDialog();
        }

        private void SigmaSlider_OnValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            currentSigma = SigmaSlider.Value;
            if (autoExecuteEnabled)
            {
                Execute();
            }
        }

        private void GaussianCheckbox_OnClick(object sender, RoutedEventArgs e)
        {
            gaussianEnabled = GaussianCheckbox.IsChecked ?? false;
            if (autoExecuteEnabled)
            {
                Execute();
            }
        }

        private void MaximumSupressionChecbox_OnClick(object sender, RoutedEventArgs e)
        {
            maxSupressionEnabled = MaximumSupressionChecbox.IsChecked ?? false;
            if (autoExecuteEnabled)
            {
                Execute();
            }
        }

        private void AutoExecuteCheckBox_OnClick(object sender, RoutedEventArgs e)
        {
            autoExecuteEnabled = AutoExecuteCheckBox.IsChecked ?? false;
        }

        private void TresholdSlider_OnDragCompleted(object sender, DragCompletedEventArgs e)
        {
            if (autoExecuteEnabled)
            {
                Execute();
            }
        }
    }
}