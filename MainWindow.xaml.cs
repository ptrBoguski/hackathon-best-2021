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
        
        public MainWindow()
        {
            InitializeComponent();
            AttachConsole(-1);
        }

        private void ImageDropZone_OnDrop(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                var filePath = ((string[]) e.Data.GetData(DataFormats.FileDrop))[0];
                LoadFile(filePath);
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
            var name = Path.GetFileName(path);
            FileNameLabel.Content = name;
        }
    }
}