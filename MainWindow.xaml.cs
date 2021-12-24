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

        private EdgeDetector edgeDetector;
        
        public MainWindow()
        {
            InitializeComponent();
            AttachConsole(-1);
            edgeDetector = new EdgeDetector();
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
            edgeDetector.FilePath = path;
            var name = Path.GetFileName(path);
            FileNameLabel.Content = name;
        }

        private void DisplayInputImage(string path)
        {
            BitmapImage bitmap = new BitmapImage();  
            bitmap.BeginInit();  
            bitmap.UriSource = new Uri(path);  
            bitmap.EndInit();
            ImageViewer.Source = bitmap;
        }

        private void ExecuteButton_OnClick(object sender, RoutedEventArgs e)
        {
            edgeDetector.Execute();
        }
    }
}