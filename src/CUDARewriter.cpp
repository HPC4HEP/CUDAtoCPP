/*
 * CUDARewriter.cpp
 *
 *  Created on: 24/lug/2015
 *      Author: latzori
 */


#include <string>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <sstream>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"


#include "clang/AST/Decl.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PPCallbacks.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm::sys::path;

static llvm::cl::OptionCategory MatcherSampleCategory("Matcher Sample");

class KernelDeclHandler : public MatchFinder::MatchCallback {
public:
	KernelDeclHandler(Rewriter &Rewrite, Preprocessor* PP) : Rewrite(Rewrite), PP(PP){}

	virtual void run(const MatchFinder::MatchResult &Result){
		clang::SourceManager* const SM = Result.SourceManager;
		if (const clang::FunctionDecl * kerneldecl = Result.Nodes.getNodeAs<clang::FunctionDecl>("CUDA_kernel_functionDecl")){
			std::string SStr;
			llvm::raw_string_ostream S(SStr);
			kerneldecl->print(S);
			//std::cout << "kerneldecl is " << S.str() << "\n";
			if(CUDAGlobalAttr* gattr = kerneldecl->getAttr<CUDAGlobalAttr>()){
				SourceLocation gattrloc = SM->getExpansionLoc(gattr->getLocation());
				if(SM->getFileID(gattrloc) != SM->getMainFileID()){
					//std::cout << "Skipped rewriting @ loc " << gattrloc.printToString(*SM) << ": FileID different from MainFileID\n";
					//std::cout << S.str() << "\n";
				} else {
					SourceLocation gattrend = PP->getLocForEndOfToken(gattrloc);
					SourceRange sr(gattrloc, gattrend);
					Rewrite.ReplaceText(gattrloc, Rewrite.getRangeSize(sr),	"");
				}
			}
		}
	}
private:
	Rewriter &Rewrite;
	Preprocessor* PP;
};

class AttributeHandler : public MatchFinder::MatchCallback {
public:
	AttributeHandler(Rewriter &Rewrite, Preprocessor* PP) : Rewrite(Rewrite), PP(PP){}

	virtual void run(const MatchFinder::MatchResult &Result){
		clang::SourceManager* const SM = Result.SourceManager;

		//Deleting __host__ attribute
		if(const clang::FunctionDecl * hostfunc = Result.Nodes.getNodeAs<clang::FunctionDecl>("CUDAHost_Attr_functionDecl")){

 			std::string SStr;
			llvm::raw_string_ostream S(SStr);
			hostfunc->print(S);
			//std::cout << "hostfunc is " << S.str() << "\n";
			if(CUDAHostAttr* hattr = hostfunc->getAttr<CUDAHostAttr>()){
				SourceLocation hattrloc = SM->getExpansionLoc(hattr->getLocation());
				if(SM->getFileID(hattrloc) != SM->getMainFileID()){
					//std::cout << "Skipped rewriting @ loc " << hattrloc.printToString(*SM) << ": FileID different from MainFileID\n";
					//std::cout << S.str() << "\n";
				} else {
					SourceLocation hattrend = PP->getLocForEndOfToken(hattrloc);
					SourceRange sr(hattrloc, hattrend);
					Rewrite.ReplaceText(hattrloc, Rewrite.getRangeSize(sr),	"");
				}
			}
		}

		//Deleting __device__ attribute
		if (const clang::FunctionDecl * devicefunc = Result.Nodes.getNodeAs<clang::FunctionDecl>("CUDADevice_Attr_functionDecl")){
			std::string SStr2;
			llvm::raw_string_ostream S2(SStr2);
			devicefunc->print(S2);
			//std::cout << "devicefunc is " << S2.str() << "\n";

			if(CUDADeviceAttr* dattr = devicefunc->getAttr<CUDADeviceAttr>()){
				SourceLocation dattrloc = SM->getExpansionLoc(dattr->getLocation());

				if(SM->getFileID(dattrloc) != SM->getMainFileID()){
					//std::cout << "Skipped rewriting @ loc " << dattrloc.printToString(*SM) << ": FileID different from MainFileID\n";
					//std::cout << S2.str() << "\n";
				} else {
					SourceLocation dattrend = PP->getLocForEndOfToken(dattrloc);
					SourceRange sr(dattrloc, dattrend);
					Rewrite.ReplaceText(dattrloc, Rewrite.getRangeSize(sr),	"");
				}
			}
		}

		//And now we have to translate the body of the function!

	}

private:
	Rewriter &Rewrite;
	Preprocessor *PP;
};

class CUDAKCallHandler : public MatchFinder::MatchCallback {
public:
	CUDAKCallHandler(Rewriter &Rewrite, Preprocessor* PP) : Rewrite(Rewrite), PP(PP){}
    //TODO: support the shared and stream exec-config parameters
	virtual void run(const MatchFinder::MatchResult &Result){
		clang::SourceManager* const SM = Result.SourceManager;

		//Do i need the check on the file ID in this case?
		if(const clang::CUDAKernelCallExpr * kernelCall = Result.Nodes.getNodeAs<clang::CUDAKernelCallExpr>("CUDA_kernel_callExpr")){

			std::cout << "\n\nDEBUG: found a kernel call!\n";

			//Name of the kernel function
			const FunctionDecl* dircallee = kernelCall->getDirectCallee(); //Refers to the kernel def!

			std::cout << "getDirectCallee: " << dircallee->getNameAsString();
			std::cout << " (starts at loc " << dircallee->getLocStart().printToString(*SM) << " )\n";

			//The kernel config is the <<<,>>> stuff
	        const CallExpr *kernelConfig = kernelCall->getConfig();
	        const Expr* grid = kernelConfig->getArg(0);
	        const Expr* block = kernelConfig->getArg(1);

	        //TEST Rewrite the threadblock expression
	        const CXXConstructExpr *construct = dyn_cast<CXXConstructExpr>(block);
	        const ImplicitCastExpr *cast = dyn_cast<ImplicitCastExpr>(construct->getArg(0));

	        //TODO: Check if all kernel launch parameters now show up as MaterializeTemporaryExpr
	    	// if so, standardize it as this with the ImplicitCastExpr fallback
	    	if (cast == NULL) {
	    		std::cout << "cast1 == NULL";
	    	    //try chewing it up as a MaterializeTemporaryExpr
	    	    const MaterializeTemporaryExpr *mat = dyn_cast<MaterializeTemporaryExpr>(construct->getArg(0));
	    	    if (mat) {
	    	    	std::cout << ", mat != NULL";
	    	    	cast = dyn_cast<ImplicitCastExpr>(mat->GetTemporaryExpr());
	    	    } else {
	    	    	std::cout << ", mat == NULL";
	    	    }
	    	    std::cout << "\n";
	    	} else {
	    		std::cout << "cast1 != NULL\n";
	    	}
	    	const DeclRefExpr *dre;
			if (cast == NULL) {
				std::cout << "cast2 == NULL, construct starts at loc "
						  << construct->getLocStart().printToString(*SM)
						  << "\n";
				dre = dyn_cast<DeclRefExpr>(construct->getArg(0));
			} else {
				std::cout << "cast2 != NULL, construct starts at loc "
						  << construct->getLocStart().printToString(*SM)
						  << "\n";
				dre = dyn_cast<DeclRefExpr>(cast->getSubExprAsWritten());
			}
			//Check if is something declared in advance or not!
			if (dre) {
				std::cout << "dre != NULL, ";
				//Variable passed
				const ValueDecl *value = dre->getDecl();
				std::string type = value->getType().getAsString();
				std::cout << "value's type is "<< type << "\n";
				unsigned int dims = 1;
				std::stringstream args;
				//TODO: just test if including cuda runtime libraries it works differently than defining my own header!!!
				if (type == "struct dim3") {
					dims = 3;
					for (unsigned int i = 0; i < 3; i++)
						std::cout << "value[" << i << "] = " << value->getNameAsString() << " | "<< value->getName().str() << "\n\n";

						//args << "localWorkSize[" << i << "] = " << value->getNameAsString() << "[" << i << "];\n";
				} else {
					//Some integer type, likely
				    SourceLocation a(SM->getExpansionLoc(dre->getLocStart())), b(Lexer::getLocForEndOfToken(SourceLocation(SM->getExpansionLoc(dre->getLocEnd())), 0,  *SM, Result.Context->getLangOpts()));
				    std::string boh = std::string(SM->getCharacterData(a), SM->getCharacterData(b)-SM->getCharacterData(a));
					std::cout << "value is "<< boh << "\n\n";
				    //args << "localWorkSize[0] = " << boh << ";\n";
				}
				//std::cout << args.str() << "\n";
			} else {
				//Some other expression passed to block
				const Expr *arg = cast->getSubExprAsWritten();
				//std::string s;
				std::cout << "dre == NULL";
				std::cout << " (arg starts at loc: " << arg->getExprLoc().printToString(*SM) << ")\n\n";
//				RewriteHostExpr(arg, s);
			}
//			const CallExpr* conf = kcall->getConfig();
//			printf("kernel call nargs %d\n", conf->getNumArgs());
//			const Expr ** args;
//			args = malloc(conf->getNumArgs()*sizeof(const Expr*));
//			for(int i = 0; i < conf->getNumArgs(); i++){
//				args[i] = conf->getArg(i);
//
//				conf->
//			}

			//PARAMETERS:
			/*
			 * we want to move blocksize, blockidx & gridsize from the cuda kernel config to both the formal (new) kernel definition and actual
			 * (new) kernel call parameters.
			 *
			 * Is it possible to get the function definition from the kernel call?
			 * is it needed actually? should we find a standard for the parameter positions?
			 * example:
			 *
			 * CUDA:
			 * __global__ void ker(T a, Tb){}
			 * ...
			 * ker<<<grid, block>>>(aa, bb);
			 *
			 * becomes
			 * C++:
			 * void ker(T a, T b, dim3 gridSize, dim3 blockSize, dim3 blockIdx){}
			 * ...
			 * ker(aa, bb, grid, block, ???);
			 *
			 */
		}
	}
private:
	Rewriter &Rewrite;
	Preprocessor *PP;

};
/**
*
* Our attempt is to write only matchers for entry points, and then
* recursively call more specific functions from the handlers!
*
* TODO list:
* - definizioni di kernel (dopo traduco il body)
* - definizioni di funzioni con attributi __device__ o __host__ (rimuovo
* 	l'attributo e poi traduco il body)
* - il main come lo tratto? perché dentro avrò sintassi cuda...
*
*/
class MyASTConsumer : public ASTConsumer {
public:
	MyASTConsumer(CompilerInstance *comp, Rewriter &R) : CI(comp)  {

		//Preprocessor
		Preprocessor* P = &CI->getPreprocessor();

		//Initialize all the handlers!
		AH = new AttributeHandler(R, P);
		KDH = new KernelDeclHandler(R, P);
		KCH = new CUDAKCallHandler(R, P);

		//Matches all the function declarations with an __host__ attribute
		Matcher.addMatcher(
				functionDecl(
						hasAttr(
								clang::attr::CUDAHost
								)
						).bind("CUDAHost_Attr_functionDecl"),
				AH);

		//Matches all the function declarations with a __device__ attribute
		Matcher.addMatcher(
				functionDecl(
						hasAttr(
								clang::attr::CUDADevice
								)
						).bind("CUDADevice_Attr_functionDecl"),
				AH);

		//Matches all the function declarations with a __global__ attribute (kernels?)
		Matcher.addMatcher(
				functionDecl(
						hasAttr(
								clang::attr::CUDAGlobal
								)
						).bind("CUDA_kernel_functionDecl"),
				KDH);

		Matcher.addMatcher(clang::ast_matchers::CUDAKernelCallExpr().bind("CUDA_kernel_callExpr"),KCH);

		//DEBUG
		//printf("ASTConsumer: added matchers\n");

	}

	// Run the matchers when we have the whole TU parsed.
	void HandleTranslationUnit(ASTContext &Context) override {
		Matcher.matchAST(Context);

	}

	//Triggered for each globally scoped declaration.
	virtual bool HandleTopLevelDecl(DeclGroupRef DG) {
		Decl *firstDecl = DG.isSingleDecl() ? DG.getSingleDecl() : DG.getDeclGroup()[0];
		SourceLocation loc = firstDecl->getLocation();
		if(CI->getSourceManager().getFileID(loc) != CI->getSourceManager().getMainFileID()){
			std::cout << "Skipping file " << loc.printToString(CI->getSourceManager()) << "\n";
			return true;
		}
		std::cout << loc.printToString(CI->getSourceManager()) << " --> Action required!\n";
		return true;
	}

private:
	MatchFinder Matcher;

	AttributeHandler *AH;

	KernelDeclHandler *KDH;

	CUDAKCallHandler *KCH;

	//Preprocessor* PP;
	CompilerInstance *CI;

};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {}
  void EndSourceFileAction() override {
    TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID())
        .write(llvm::outs());
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<MyASTConsumer>(&CI, TheRewriter);
  }

private:
  Rewriter TheRewriter;
};

int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, MatcherSampleCategory);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());

  return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}

