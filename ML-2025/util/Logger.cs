using System;
using System.IO;
using System.Threading.Tasks;

namespace ML_2025.Helpers
{
    public static class Logger
    {
        public static async Task SalvarLogAsync(string userInput, string aiResponse)
        {
      
            string pastaProjeto = Directory.GetParent(AppDomain.CurrentDomain.BaseDirectory)!.Parent!.FullName;
            string pastaLogs = Path.Combine(pastaProjeto, "Logs");

            if (!Directory.Exists(pastaLogs))
                Directory.CreateDirectory(pastaLogs);


            string arquivo = Path.Combine(pastaLogs, $"log_{DateTime.Now:yyyy-MM-dd}.txt");

            string logText =
$@"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}]
Usuário: {userInput}
IA: {aiResponse}
---------------------------------------

";

            await File.AppendAllTextAsync(arquivo, logText);

            Console.WriteLine($"[Logger] Log salvo: {arquivo}");
        }
    }
}
