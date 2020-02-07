// Fill out your copyright notice in the Description page of Project Settings.

using System.IO;
using UnrealBuildTool;

public class IAF : ModuleRules
{
    private string ModulePath
    {
        get { return ModuleDirectory; }
    }

    private string ThirdPartyPath
    {
        get { return Path.GetFullPath(Path.Combine(ModulePath, "../../ThirdParty/")); }
    }
    public IAF(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
	
		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore" });

		PrivateDependencyModuleNames.AddRange(new string[] {  });

        // Uncomment if you are using Slate UI
        // PrivateDependencyModuleNames.AddRange(new string[] { "Slate", "SlateCore" });

        // Uncomment if you are using online features
        // PrivateDependencyModuleNames.Add("OnlineSubsystem");

        // To include OnlineSubsystemSteam, add it to the plugins section in your uproject file with the Enabled attribute set to true

        //Load libs
        bEnableUndefinedIdentifierWarnings = false;
        bool sup = LoadTorchLibs(Target);
    }

    public bool LoadTorchLibs(ReadOnlyTargetRules Target)
    {
        //Type = ModuleType.External;
        bool isLibrarySupported = false;

        if ((Target.Platform == UnrealTargetPlatform.Win64) || (Target.Platform == UnrealTargetPlatform.Win32))
        {
            isLibrarySupported = true;

            string PlatformString = (Target.Platform == UnrealTargetPlatform.Win64) ? "x64" : "x86";
            string LibrariesPath = Path.Combine(ThirdPartyPath, "libtorch", "lib");
            // Include path
            PublicIncludePaths.Add(Path.Combine(ThirdPartyPath, "libtorch", "include"));
            PublicIncludePaths.Add(Path.Combine(ThirdPartyPath, "libtorch", "include", "torch", "csrc", "api", "include"));

            /*
            test your path with:
            using System; // Console.WriteLine("");
            Console.WriteLine("... LibrariesPath -> " + LibrariesPath);
            */

            //PublicAdditionalLibraries.Add(Path.Combine(LibrariesPath, "BobsMagic." + PlatformString + ".lib"));
            PublicAdditionalLibraries.Add(Path.Combine(LibrariesPath, "torch.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(LibrariesPath, "c10.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(LibrariesPath, "c10_cuda.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(LibrariesPath, "caffe2_module_test_dynamic.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(LibrariesPath, "caffe2_nvrtc.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(LibrariesPath, "clog.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(LibrariesPath, "cpuinfo.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(LibrariesPath, "libprotobuf.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(LibrariesPath, "libprotobuf-lite.lib"));
            PublicAdditionalLibraries.Add(Path.Combine(LibrariesPath, "libprotoc.lib"));

            PublicLibraryPaths.Add(LibrariesPath);

            PublicDelayLoadDLLs.AddRange(new string[] { "torch.dll" });
        }

        if (isLibrarySupported)
        {
            //// Include path
            //PublicIncludePaths.Add(Path.Combine(ThirdPartyPath, "libtorch", "include"));
        }

        PublicDefinitions.Add(string.Format("WITH_LIBTORCH_PPO_BINDING={0}", isLibrarySupported ? 1 : 0));

        return isLibrarySupported;
    }
}
